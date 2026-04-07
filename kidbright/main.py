import math
import network
import ujson
from machine import Pin, ADC, UART
from time import sleep, sleep_ms
from umqtt.robust import MQTTClient
import dht

from config import (
    WIFI_SSID, WIFI_PASS,
    MQTT_BROKER, MQTT_USER, MQTT_PASS
)

# =====================================
# PROJECT CONFIG
# =====================================
DEVICE_ID = "SleepSense_01"

# collect every 2 minutes
SAMPLE_INTERVAL_SEC = 120

# aggregate 15 readings -> publish every 20 minutes
BATCH_SIZE = 10

# MQTT topic
MQTT_TOPIC = "{}/sleepsense/{}".format(MQTT_USER, DEVICE_ID)

# =====================================
# PIN CONFIG
# =====================================
DHT_PIN = 32
LIGHT_PIN = 36
SOUND_PIN = 34

# PMS7003
PMS_TX_PIN = 23   # board TX -> PMS RX
PMS_RX_PIN = 19   # board RX <- PMS TX
USE_DUST = True

# =====================================
# LIGHT CALIBRATION
# =====================================
def raw_to_lux(raw):
    if raw is None:
        return None

    raw = max(0, min(raw, 850))
    normalized = (850 - raw) / 850
    gamma = 1.5
    lux = (normalized ** gamma) * 1000
    return round(lux, 2)

# =====================================
# SOUND CALIBRATION
# =====================================
ADC_MAX = 4095
SOUND_SAMPLE_COUNT = 120
SMOOTH_ALPHA = 0.25

DB_A = 21.65
DB_B = -2.53

MIN_RAW_FOR_DB = 1
MIN_DB = 35
MAX_DB = 90

SND_EVENT_THRESHOLD_DB = 55

smoothed_db = 0

# =====================================
# HARDWARE INIT
# =====================================
led_wifi = Pin(2, Pin.OUT)
led_iot = Pin(12, Pin.OUT)

led_wifi.value(1)
led_iot.value(1)

dht_sensor = dht.DHT11(Pin(DHT_PIN))

ldr = ADC(Pin(LIGHT_PIN))
ldr.atten(ADC.ATTN_11DB)
ldr.width(ADC.WIDTH_12BIT)

sound_adc = ADC(Pin(SOUND_PIN))
try:
    sound_adc.atten(ADC.ATTN_11DB)
except:
    pass

try:
    sound_adc.width(ADC.WIDTH_12BIT)
except:
    pass

if USE_DUST:
    pms_uart = UART(2, baudrate=9600, tx=Pin(PMS_TX_PIN), rx=Pin(PMS_RX_PIN))
    last_dust = {"pm1": None, "pm25": None, "pm10": None}
else:
    last_dust = None

# =====================================
# WIFI / MQTT
# =====================================
wlan = None
mqtt = None

def connect_wifi():
    global wlan

    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if not wlan.isconnected():
        print("Connecting to WiFi...")
        wlan.connect(WIFI_SSID, WIFI_PASS)

        timeout = 20
        while (not wlan.isconnected()) and timeout > 0:
            sleep(1)
            timeout -= 1

    if not wlan.isconnected():
        led_wifi.value(1)
        raise Exception("WiFi connection failed")

    led_wifi.value(0)
    print("WiFi connected:", wlan.ifconfig())

def ensure_wifi():
    global wlan
    if wlan is None or not wlan.isconnected():
        connect_wifi()

def connect_mqtt():
    global mqtt

    print("Connecting to MQTT...")
    mqtt = MQTTClient(
        client_id=DEVICE_ID,
        server=MQTT_BROKER,
        user=MQTT_USER,
        password=MQTT_PASS
    )
    mqtt.connect()
    led_iot.value(0)
    print("MQTT connected")

def ensure_mqtt():
    global mqtt

    if mqtt is None:
        connect_mqtt()
        return

    try:
        mqtt.ping()
    except:
        try:
            mqtt.disconnect()
        except:
            pass
        mqtt = None
        connect_mqtt()

# =====================================
# SENSOR FUNCTIONS
# =====================================
def read_dht():
    try:
        dht_sensor.measure()
        temp = dht_sensor.temperature()
        hum = dht_sensor.humidity()
        return round(temp, 2), round(hum, 2)
    except OSError as e:
        print("DHT11 read error:", e)
        return None, None

def read_light_raw():
    try:
        return ldr.read()
    except Exception as e:
        print("Light read error:", e)
        return None

def read_light_lux():
    raw = read_light_raw()
    return raw_to_lux(raw)

def read_sound_p2p(sample_count=SOUND_SAMPLE_COUNT):
    min_val = ADC_MAX
    max_val = 0

    for _ in range(sample_count):
        v = sound_adc.read()
        if v < min_val:
            min_val = v
        if v > max_val:
            max_val = v

    return max_val - min_val

def raw_to_db(raw_p2p):
    if raw_p2p is None:
        return None

    raw_p2p = max(MIN_RAW_FOR_DB, raw_p2p)
    db = (DB_A * math.log(raw_p2p, 10)) + DB_B

    if db < MIN_DB:
        db = MIN_DB
    elif db > MAX_DB:
        db = MAX_DB

    return round(db, 2)

def get_sound_stats(window_count=5):
    global smoothed_db

    db_list = []

    for _ in range(window_count):
        raw_p2p = read_sound_p2p()
        db = raw_to_db(raw_p2p)

        smoothed_db = (SMOOTH_ALPHA * db) + ((1 - SMOOTH_ALPHA) * smoothed_db)
        db_list.append(round(smoothed_db, 2))

        sleep_ms(200)

    if len(db_list) == 0:
        return None, None, None, None

    snd_avg = sum(db_list) / len(db_list)
    snd_peak = max(db_list)

    mean_db = snd_avg
    variance = 0
    for v in db_list:
        variance += (v - mean_db) ** 2
    variance = variance / len(db_list)
    snd_var = math.sqrt(variance)

    snd_evt = 0
    for v in db_list:
        if v >= SND_EVENT_THRESHOLD_DB:
            snd_evt += 1

    return round(snd_avg, 2), round(snd_peak, 2), round(snd_var, 2), snd_evt

def read_pms7003():
    global last_dust

    if not USE_DUST:
        return {"pm1": None, "pm25": None, "pm10": None}

    try:
        n = pms_uart.any()
        if n < 32:
            return last_dust

        buf = pms_uart.read()
        if not buf or len(buf) < 32:
            return last_dust

        for i in range(len(buf) - 31):
            if buf[i] == 0x42 and buf[i + 1] == 0x4D:
                frame = buf[i:i + 32]

                pm1 = frame[10] * 256 + frame[11]
                pm25 = frame[12] * 256 + frame[13]
                pm10 = frame[14] * 256 + frame[15]

                last_dust = {
                    "pm1": round(pm1, 2),
                    "pm25": round(pm25, 2),
                    "pm10": round(pm10, 2)
                }
                return last_dust

        return last_dust

    except Exception as e:
        print("PMS7003 read error:", e)
        return last_dust

# =====================================
# COLLECT + AGGREGATE
# =====================================
def collect_processed_reading():
    temp_c, hum_pct = read_dht()
    light_lux = read_light_lux()
    snd_avg, snd_peak, snd_var, snd_evt = get_sound_stats()
    dust = read_pms7003()

    return {
        "temp_c": temp_c,
        "hum_pct": hum_pct,
        "light_lux": light_lux,
        "snd_avg": snd_avg,
        "snd_peak": snd_peak,
        "snd_var": snd_var,
        "snd_evt": snd_evt,
        "pm1": dust["pm1"] if dust else None,
        "pm25": dust["pm25"] if dust else None,
        "pm10": dust["pm10"] if dust else None
    }

def avg_key(items, key):
    vals = [x[key] for x in items if x.get(key) is not None]
    if len(vals) == 0:
        return None
    return round(sum(vals) / len(vals), 2)

def max_key(items, key):
    vals = [x[key] for x in items if x.get(key) is not None]
    if len(vals) == 0:
        return None
    return round(max(vals), 2)

def sum_key(items, key):
    vals = [x[key] for x in items if x.get(key) is not None]
    if len(vals) == 0:
        return None
    return int(sum(vals))

def aggregate_batch(batch):
    return {
        "device_id": "SleepSense_01",
        "temp_c": avg_key(batch, "temp_c"),
        "hum_pct": avg_key(batch, "hum_pct"),
        "light_lux": avg_key(batch, "light_lux"),
        "snd_avg": avg_key(batch, "snd_avg"),
        "snd_peak": max_key(batch, "snd_peak"),
        "snd_var": avg_key(batch, "snd_var"),
        "snd_evt": sum_key(batch, "snd_evt"),
        "pm1": avg_key(batch, "pm1"),
        "pm25": avg_key(batch, "pm25"),
        "pm10": avg_key(batch, "pm10")
    }

# =====================================
# MQTT PUBLISH
# =====================================
def publish_payload(payload):
    ensure_wifi()
    ensure_mqtt()

    body = ujson.dumps(payload)
    mqtt.publish(MQTT_TOPIC, body)

    print("Published to:", MQTT_TOPIC)
    print(body)
    print("-" * 40)

# =====================================
# MAIN LOOP
# =====================================
print("SleepSense started")
print("Device:", DEVICE_ID)
print("MQTT topic:", MQTT_TOPIC)

if USE_DUST:
    print("Warming up PMS7003...")
    sleep(10)

batch = []

while True:
    try:
        reading = collect_processed_reading()
        batch.append(reading)

        print("Reading {}/{} collected".format(len(batch), BATCH_SIZE))

        if len(batch) >= BATCH_SIZE:
            payload = aggregate_batch(batch)
            publish_payload(payload)
            batch = []

        sleep(SAMPLE_INTERVAL_SEC)

    except Exception as e:
        print("Main loop error:", e)
        led_iot.value(1)
        sleep(5)