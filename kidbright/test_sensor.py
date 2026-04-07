import math
from machine import Pin, ADC, UART
from time import sleep, sleep_ms
import dht

# =====================================
# PROJECT CONFIG
# =====================================
DEVICE_ID = "SleepSense_01"

# Read every 2 seconds
SAMPLE_INTERVAL_SEC = 0.2

# Summarize every 10 seconds = 5 readings
BATCH_SIZE = 5

# =====================================
# PIN CONFIG
# =====================================
DHT_PIN = 32
LIGHT_PIN = 33
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

    # clamp
    raw = max(0, min(raw, 4095))

    # calibration points
    # raw = 0    -> 15000 lux
    # raw = 850  -> 250 lux
    # raw = 2500 -> 35 lux   (middle of your observed 20–50 lux range)
    # raw = 4095 -> 0 lux

    # Segment 1: bright range (0 -> 850)
    # use log interpolation to handle large lux drop smoothly
    if raw <= 850:
        x1, y1 = 0, 15000
        x2, y2 = 850, 250

        t = (raw - x1) / (x2 - x1)
        log_y1 = math.log10(y1)
        log_y2 = math.log10(y2)
        log_lux = log_y1 + t * (log_y2 - log_y1)
        lux = 10 ** log_lux

    # Segment 2: medium-to-low range (850 -> 2500)
    # also use log interpolation so 2500 still stays around 20–50 lux
    elif raw <= 2500:
        x1, y1 = 850, 250
        x2, y2 = 2500, 35

        t = (raw - x1) / (x2 - x1)
        log_y1 = math.log10(y1)
        log_y2 = math.log10(y2)
        log_lux = log_y1 + t * (log_y2 - log_y1)
        lux = 10 ** log_lux

    # Segment 3: very dark range (2500 -> 4095)
    # fade down linearly to zero
    else:
        x1, y1 = 2500, 35
        x2, y2 = 4095, 0

        t = (raw - x1) / (x2 - x1)
        lux = y1 + t * (y2 - y1)

    return round(max(lux, 0), 2)

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
# SENSOR READ FUNCTIONS
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


# =====================================
# DUST
# =====================================
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
# COLLECT ONE PROCESSED READING
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


# =====================================
# AGGREGATION
# =====================================
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
        "device_id": DEVICE_ID,
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
# PRINT HELPERS
# =====================================
def print_summary(payload):
    print("\n" + "=" * 58)
    print("           SLEEP SENSE SUMMARY (10 SECONDS)")
    print("=" * 58)
    print("Device             : {}".format(payload["device_id"]))
    print("Temp               : {} °C".format(payload["temp_c"]))
    print("Humidity           : {} %".format(payload["hum_pct"]))
    print("Light              : {} lux".format(payload["light_lux"]))
    print("Sound Avg          : {} dB est.".format(payload["snd_avg"]))
    print("Sound Peak         : {} dB est.".format(payload["snd_peak"]))
    print("Sound Variability  : {} dB".format(payload["snd_var"]))
    print("Sound Events       : {} count".format(payload["snd_evt"]))
    print("PM1.0              : {} µg/m³".format(payload["pm1"]))
    print("PM2.5              : {} µg/m³".format(payload["pm25"]))
    print("PM10               : {} µg/m³".format(payload["pm10"]))
    print("=" * 58)


# =====================================
# MAIN LOOP
# =====================================
print("SleepSense test started")
print("Collect every {} sec".format(SAMPLE_INTERVAL_SEC))
print("Summarize every {} sec".format(SAMPLE_INTERVAL_SEC * BATCH_SIZE))

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
            summary = aggregate_batch(batch)
            print_summary(summary)
            batch = []

        sleep(SAMPLE_INTERVAL_SEC)

    except Exception as e:
        print("Main loop error:", e)
        sleep(3)