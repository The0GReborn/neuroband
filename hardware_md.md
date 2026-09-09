# NeuroBand BCI — Complete Hardware List

**Low-cost brain-to-speech interface**
**Total cost: ~₹9,230**
**Prices verified: April 2026 · Indian sources · Excludes shipping**

---

## Signal Path (Scalp → Speaker)

```
Brain activity
  → Scalp electrodes (F3, F4, A1, Fpz)
  → BioAmp cable V3
  → BioAmp EXG Pill ×2  (amplify 10µV → 1V)
  → CS1237 24-bit ADC ×2  (analog → 24-bit digital @ 250 Hz)
  → Raspberry Pi Zero 2 W  (filter → features → template match)
  → USB audio dongle
  → Mini speaker  (speaks detected word aloud)
```

---

## 1 · Brain Signal Capture (Electrodes)

### Item 01 — EEG Gel Electrodes (24 pcs)
| Field | Detail |
|---|---|
| **Category** | Electrodes |
| **Quantity** | 24 pcs (included in Explorer Pack bundle) |
| **Price** | Bundled free with item 03 |
| **Function** | Makes electrical contact with scalp skin to pick up brain signals (10–100 µV) |
| **Placement** | F3 (left forehead), F4 (right forehead), A1 (left earlobe), Fpz (center hairline — ground) |
| **Connects to** | BioAmp cable → BioAmp EXG Pill IN+ and IN− pins |
| **Buy URL** | Included in Explorer Pack — no separate purchase needed |

---

### Item 02 — 2-Channel EEG Headband (dry electrodes)
| Field | Detail |
|---|---|
| **Category** | Electrodes |
| **Quantity** | 1 (included in Explorer Pack bundle) |
| **Price** | Bundled free with item 03 |
| **Function** | Stretchable band holds dry electrodes on forehead for quick testing without gel |
| **When to use** | Testing and development only. Use gel electrodes for real training data collection |
| **Connects to** | BioAmp cable V3 → BioAmp EXG Pill |
| **Buy URL** | Included in Explorer Pack — no separate purchase needed |

---

## 2 · Signal Amplification

### Item 03 — BioAmp EXG Pill Explorer Pack (×2 assembled) ⭐ KEY ITEM
| Field | Detail |
|---|---|
| **Category** | Amplifier |
| **Quantity** | 1 pack (contains 2 assembled pills) |
| **Price** | ~₹4,600 |
| **Function** | Analog front-end amplifier. Boosts tiny brain signals (10 µV) to a voltage the ADC can read. One pill per EEG channel (F3-A1 and F4-A1) |
| **Connects to** | Input: scalp electrodes via BioAmp cable. Output: OUT pin → CS1237 ADC analog input. Power: 3.3V + GND from Pi GPIO |
| **Bundle includes** | 2× assembled BioAmp pills · 24 gel electrodes · BioAmp cable V3 · 2-ch EEG headband · 6 jumper cables · 8 zip ties |
| **Buy URL** | https://robu.in/product/bioamp-exg-pill-unassembled-exg-explorer-pack-2/ (SKU: R182369) |
| **Backup URL** | https://www.amazon.in/Explorer-Neuroscience-Upside-Down-Labs/dp/B0B29CCPQB |

---

## 3 · Analog-to-Digital Conversion

### Item 04 — xcluma CS1237 24-bit ADC Module (×2)
| Field | Detail |
|---|---|
| **Category** | ADC |
| **Quantity** | 2 modules |
| **Price** | ₹414 × 2 = ₹828 |
| **Function** | Converts amplified analog EEG voltage into 24-bit digital numbers at 250 samples/sec that the Pi can process. One module per channel |
| **Connects to** | Input: BioAmp OUT pin. Output: GPIO11 (shared SCLK), GPIO9 (ch0 F3 data), GPIO10 (ch1 F4 data). Power: 3.3V + GND from Pi |
| **Specs** | 24-bit resolution · PGA gain 128 · 2-wire SPI · TL431 onboard voltage reference · −25°C to 85°C |
| **Buy URL** | https://www.amazon.in/xcluma-Onboard-External-Reference-Single-channel/dp/B0D8S1VGQ4 |

---

## 4 · Processing Unit

### Item 05 — Raspberry Pi Zero 2 W ⭐ KEY ITEM
| Field | Detail |
|---|---|
| **Category** | Processor |
| **Quantity** | 1 |
| **Price** | ₹1,922 |
| **Function** | Main computer. Runs all Python code: signal filtering (Butterworth/Notch), feature extraction (25 EEG features), template matching classifier, audio output, and web server for the phone training app |
| **Connects to** | CS1237 ADC via GPIO pins. USB audio dongle via OTG adapter. Battery via micro-USB power port. Phone via built-in WiFi (hotspot mode) |
| **GPIO pins used** | GPIO9 (DOUT ch0 F3), GPIO10 (DOUT ch1 F4), GPIO11 (shared SCLK), 3.3V, GND |
| **Specs** | Quad-core ARM Cortex-A53 @ 1GHz · 512MB RAM · 2.4GHz WiFi · Bluetooth 4.2 · 40-pin GPIO · 65mm × 30mm |
| **Buy URL** | https://robocraze.com/products/raspberry-pi-zero-2-w (authorised seller) |

---

### Item 06 — microSD Card 16GB Class 10
| Field | Detail |
|---|---|
| **Category** | Storage |
| **Quantity** | 1 |
| **Price** | ~₹200 |
| **Function** | Stores Raspberry Pi OS, all Python scripts, trained EEG word templates (.pkl files), and EEG session logs |
| **Connects to** | Slides into Pi Zero 2 W microSD slot. Must be flashed with Raspberry Pi OS Lite using Raspberry Pi Imager before first use |
| **Buy URL** | https://www.amazon.in/s?k=sandisk+microsd+16gb+class+10 |

---

## 5 · Audio Output

### Item 07 — USB Audio Dongle (USB to 3.5mm)
| Field | Detail |
|---|---|
| **Category** | Audio |
| **Quantity** | 1 |
| **Price** | ~₹150 |
| **Function** | Pi Zero 2 W has no built-in 3.5mm audio jack. This USB dongle adds one so the system can play spoken word audio (YES, NO, HELLO, THANK YOU, etc.) |
| **Connects to** | Pi micro-USB OTG port via OTG adapter (item 08). Speaker plugs into dongle's 3.5mm jack. Plug-and-play, no driver needed on Linux |
| **Buy URL** | https://www.amazon.in/s?k=usb+audio+dongle+sound+card+3.5mm |

---

### Item 08 — Micro-USB OTG Adapter
| Field | Detail |
|---|---|
| **Category** | Adapter |
| **Quantity** | 1 |
| **Price** | ~₹80 |
| **Function** | Converts Pi's micro-USB OTG port into a full-size USB-A female port so the USB audio dongle can plug in |
| **Connects to** | Pi micro-USB OTG port on one side. USB audio dongle plugs into USB-A female on the other side |
| **Buy URL** | https://robocraze.com/products/high-speed-micro-usb-otg-cable-33-cm |
| **Backup URL** | https://www.amazon.in/s?k=micro+usb+otg+adapter |

---

### Item 09 — Mini Speaker 3W (3.5mm jack)
| Field | Detail |
|---|---|
| **Category** | Speaker |
| **Quantity** | 1 |
| **Price** | ~₹300 |
| **Function** | Speaks detected words aloud — HELLO, THANK YOU, HELP, etc. Core output device of the brain-to-speech system |
| **Connects to** | 3.5mm audio jack plugs into USB audio dongle output |
| **Buy URL** | https://www.amazon.in/s?k=mini+speaker+3.5mm+portable+3w |

---

## 6 · Power System

### Item 10 — Samsung 18650 30Q Li-Ion Battery Cell (×2)
| Field | Detail |
|---|---|
| **Category** | Power |
| **Quantity** | 2 cells |
| **Price** | ~₹400 × 2 = ₹800 |
| **Function** | Rechargeable lithium-ion cells that power the entire NeuroBand system (Pi + BioAmps + ADCs + speaker) for ~3 hours per charge |
| **Connects to** | Slots into power bank module → micro-USB cable → Pi power port |
| **Specs** | 3000mAh · 3.7V nominal · 15A max discharge · Samsung SDI quality |
| **Buy URL** | https://robu.in/product/samsung-18650-30q-li-ion-battery/ |

---

### Item 11 — 18650 DIY Power Bank Module (5V USB output + charger)
| Field | Detail |
|---|---|
| **Category** | Power |
| **Quantity** | 1 |
| **Price** | ~₹200 |
| **Function** | Boost converter + charge controller. Steps up 3.7V battery to stable 5V USB output for the Pi. Also charges the 18650 cells via micro-USB input |
| **Connects to** | 18650 cells plug into B+ B− pads on board. USB output → micro-USB cable → Pi power port. Charge input via micro-USB from any phone charger |
| **Buy URL** | https://robu.in/product/18650-5v-1a-2a-lithium-battery-digital-display-charging-module-with-dual-usb-output-quick-charge-supported/ |

---

## 7 · Wiring & Connectors

### Item 12 — Jumper Wires (M-M, M-F, F-F assortment)
| Field | Detail |
|---|---|
| **Category** | Wiring |
| **Quantity** | 1 set (6 wires already in Explorer Pack; buy extra set for safety) |
| **Price** | ~₹100 |
| **Function** | Connect CS1237 ADC modules to Pi GPIO pins. 5 wires needed minimum: SCLK, DOUT ch0, DOUT ch1, 3.3V, GND |
| **Connects to** | CS1237 ADC header pins ↔ Raspberry Pi Zero 2 W GPIO header pins |
| **Buy URL** | https://robu.in/product/40-pin-dupont-jumper-wire-male-to-female-20cm/ |
| **Backup URL** | https://www.amazon.in/s?k=jumper+wires+male+female+40+pin |

---

## 8 · Consumables

### Item 13 — Alcohol Swabs (skin preparation)
| Field | Detail |
|---|---|
| **Category** | Consumable |
| **Quantity** | 1 pack |
| **Price** | ~₹50 |
| **Function** | Clean scalp and earlobe skin before placing electrodes. Removes oil to reduce electrode impedance and improve signal quality |
| **How to use** | Wipe F3, F4, A1, Fpz positions before placing electrodes. Wait 10 seconds for skin to dry before attaching |
| **Buy** | Any pharmacy — Dettol or Savlon pre-saturated swabs |

---

## Cost Summary

| # | Item | Qty | Unit ₹ | Total ₹ |
|---|---|---|---|---|
| 01 | EEG Gel Electrodes (24 pcs) | 1 pack | Bundled | — |
| 02 | 2-Channel EEG Headband | 1 | Bundled | — |
| 03 | BioAmp EXG Pill Explorer Pack ×2 | 1 pack | 4,600 | 4,600 |
| 04 | CS1237 24-bit ADC Module | ×2 | 414 | 828 |
| 05 | Raspberry Pi Zero 2 W | 1 | 1,922 | 1,922 |
| 06 | microSD Card 16GB Class 10 | 1 | 200 | 200 |
| 07 | USB Audio Dongle | 1 | 150 | 150 |
| 08 | Micro-USB OTG Adapter | 1 | 80 | 80 |
| 09 | Mini Speaker 3W | 1 | 300 | 300 |
| 10 | Samsung 18650 30Q Battery | ×2 | 400 | 800 |
| 11 | 18650 Power Bank Module | 1 | 200 | 200 |
| 12 | Jumper Wires (assortment) | 1 set | 100 | 100 |
| 13 | Alcohol Swabs | 1 pack | 50 | 50 |
| | | | **TOTAL** | **~₹9,230** |

---

## GPIO Wiring Reference

```
CS1237 #1 (F3-A1 channel)      CS1237 #2 (F4-A1 channel)
  DOUT  →  Pi GPIO 9              DOUT  →  Pi GPIO 10
  SCLK  →  Pi GPIO 11             SCLK  →  Pi GPIO 11  (shared)
  VCC   →  Pi 3.3V                VCC   →  Pi 3.3V
  GND   →  Pi GND                 GND   →  Pi GND
```

---

## Electrode Placement Reference

```
F3  — Left forehead    → BioAmp #1 IN+
F4  — Right forehead   → BioAmp #2 IN+
A1  — Left earlobe     → BioAmp #1 IN− and BioAmp #2 IN− (shared)
Fpz — Center hairline  → BioAmp #1 REF and BioAmp #2 REF (shared ground)
```

---

## Notes

- **Robocraze** is the only authorised Raspberry Pi seller in India — buy the Pi Zero 2 W only from them to guarantee genuine hardware and 1-year warranty.
- **BioAmp EXG Pill** is designed and manufactured in India by Upside Down Labs (New Delhi).
- **xcluma CS1237** is available on Amazon.in at ₹414 per unit — buy 2.
- The **USB audio dongle + OTG adapter** (items 07 + 08) are critical and easy to forget. The Pi Zero 2 W has no built-in audio jack — without these two items the speaker cannot work.
- Gel electrodes give significantly better signal quality than dry electrodes. Use the bundled gel electrodes for all training data collection sessions.
- Total is well under ₹10,000 budget. Comparable commercial BCI systems cost ₹30,000–₹5,00,000+.
