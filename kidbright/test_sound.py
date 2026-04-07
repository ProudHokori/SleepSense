import math
import network
import ujson
from machine import Pin, ADC
from time import sleep, sleep_ms
from umqtt.robust import MQTTClient

from config import (
    WIFI_SSID, WIFI_PASS,
    MQTT_BROKER, MQTT_USER, MQTT_PASS
)

# =====================================
# PROJECT CONFIG
# =====================================
DEVICE_ID = "SleepSense_01"

# publish every 200 ms
PUBLISH_INTERVAL_MS = 200

# MQTT topic
MQTT_TOPIC = "{}/sleepsense/{}".format(MQTT_USER, DEVICE_ID)

# =====================================
# PIN CONFIG
# =====================================
SOUND_PIN = 34

# =====================================
# SOUND CALIBRATION
# =====================================
ADC_MAX = 4095
SOUND_SAMPLE_COUNT = 120
SMOOTH_ALPHA = 0.25

# keep the calibrated formula exactly the same
DB_A = 21.65
DB_B = -2.53

MIN_RAW_FOR_DB = 1
MIN_DB = 35
MAX_DB = 90

smoothed_raw = 0
smoothed_db = 0

# =====================================
# HARDWARE INIT
# =====================================
led_wifi = Pin(2, Pin.OUT)
led_iot = Pin(12, Pin.OUT)

led_wifi.value(1)
led_iot.value(1)

sound_adc = ADC(Pin(SOUND_PIN))
try:
    sound_adc.atten(ADC.ATTN_11DB)
except:
    pass

try:
    sound_adc.width(ADC.WIDTH_12BIT)
except:
    pass

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
# SOUND FUNCTIONS
# =====================================
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

    # keep calibrated formula exactly as before
    db = (DB_A * math.log(raw_p2p, 10)) + DB_B

    if db < MIN_DB:
        db = MIN_DB
    elif db > MAX_DB:
        db = MAX_DB

    return round(db, 2)

def db_to_normalized(db, min_db=40, max_db=75):
    if db is None:
        return None

    if db < min_db:
        db = min_db
    elif db > max_db:
        db = max_db

    norm = (db - min_db) / (max_db - min_db)
    return round(norm, 4)

def get_sound_data():
    global smoothed_raw, smoothed_db

    raw_p2p = read_sound_p2p()
    db = raw_to_db(raw_p2p)

    smoothed_raw = (SMOOTH_ALPHA * raw_p2p) + ((1 - SMOOTH_ALPHA) * smoothed_raw)
    smoothed_db = (SMOOTH_ALPHA * db) + ((1 - SMOOTH_ALPHA) * smoothed_db)

    normalized = db_to_normalized(smoothed_db)

    return {
        "device_id": DEVICE_ID,
        "sensor": "sound",
        "raw_p2p": raw_p2p,
        "db": round(db, 2),
        "smoothed_raw": round(smoothed_raw, 2),
        "smoothed_db": round(smoothed_db, 2),
        "normalized": normalized,
        "percent": round(normalized * 100, 2)
    }

# =====================================
# MQTT PUBLISH
# =====================================
def publish_payload(payload):
    ensure_wifi()
    ensure_mqtt()

    body = ujson.dumps(payload)
    mqtt.publish(MQTT_TOPIC, body)

    print(
        "Raw:{:4d} | dB:{:6.2f} | Smooth dB:{:6.2f} | Norm:{:.4f} | {:6.2f}%".format(
            payload["raw_p2p"],
            payload["db"],
            payload["smoothed_db"],
            payload["normalized"],
            payload["percent"]
        )
    )

# =====================================
# MAIN LOOP
# =====================================
print("SleepSense realtime sound started")
print("Device:", DEVICE_ID)
print("MQTT topic:", MQTT_TOPIC)

while True:
    try:
        payload = get_sound_data()
        publish_payload(payload)
        sleep_ms(PUBLISH_INTERVAL_MS)

    except Exception as e:
        print("Main loop error:", e)
        led_iot.value(1)
        sleep(2)
