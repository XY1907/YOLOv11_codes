# SpotShrooms — IIICE 2026 Video Script (online / virtual entry)

**Spec (mandatory for virtual participants):** 6–8 minutes · **landscape** · uploaded to **YouTube (public)** ·
team members' names + "Singapore Polytechnic" stated **at the start** · cover Problem → Innovative
Solution & Demonstration → Impact & Benefits → Commercialization/Implementation.

**Target ~7:00.** Narration is written to be read at a calm pace (~140 wpm). [BRACKETS] = on-screen visual.

---

### 0:00 – 0:30 · Open  *(required: names + organisation)*
[On screen: SpotShrooms title card + team names + "Singapore Polytechnic"]

> "Hi, we're **Team SpotShrooms** from **Singapore Polytechnic** — Jordan, Sean, Xin Yin and Pasu.
> Our project is **SpotShrooms: an AI-powered system that gives mushroom farms eyes on every shelf —
> and waters them automatically, only when they need it.**"

### 0:30 – 1:30 · The Problem
[On screen: mushroom farm footage / the 20–30% stat]

> "Mushroom farming across Asia still depends on **manual monitoring**. A worker walks the racks and
> decides, by eye, what to water. But mushrooms grow around the clock, and no one can watch every shelf
> all the time. The result is **over-watering** — and once a mature mushroom sits in too much moisture,
> it rots. Farms lose **20 to 30 percent** of their harvest this way every year. That's food, water and
> income — wasted. The core problem is simple: **watering is a guess, and the guess is often wrong.**"

### 1:30 – 2:45 · Our Solution
[On screen: the 6-step pipeline diagram animating left to right]

> "SpotShrooms replaces that guess with a **self-driving watering layer**. Here's how it works.
> A camera moves along the shelves and feeds each frame to a **YOLOv8 vision model**, which reads the
> **growth stage** of every cluster — no sprout, small-and-medium, or mature — and decides **whether**
> watering is even appropriate. At the same time, **DHT22 and CO₂ sensors** read the microclimate.
> Those readings go into a **Random Forest model** that predicts **exactly how long to mist, in seconds**.
> A **Raspberry Pi** then triggers the valve — and everything shows up live on a **web dashboard**.
> Two AI models cooperating: one decides *whether* to water, the other decides *how long*. That pairing
> is what makes SpotShrooms **crop-aware** instead of a blind timer."

### 2:45 – 4:45 · Demonstration
[On screen: SCREEN-RECORD the live dashboard — Live Monitoring tab, then Irrigation Timing tab; then the prototype + circuit]

> "Let me show you the working system. This is our dashboard. On **Live Monitoring**, each shelf is
> detected in real time — green boxes for water, red for don't-water — with temperature and humidity
> gauges against the safe band. Switch to **Irrigation Timing**, and the system gives a clear
> **irrigate-now or wait** verdict, the recommended watering window, and the reason — here it's holding
> off because the mushrooms are mature. That's the safety guardrail in action: **it will never water a
> mature shelf**, which is exactly where the spoilage comes from.
> [show prototype] Physically, it's a camera on a moving rod over the shelves, a DHT22 and CO₂ sensor,
> and a Raspberry Pi driving the valve — all of which you can see in our prototype and wiring."

### 4:45 – 5:45 · Results & Validation
[On screen: the metric cards + confusion matrix + actual-vs-predicted chart]

> "And it works. Our **YOLOv8** detector reaches **0.97 precision** — meaning when it flags a mushroom,
> it's almost always right, so we don't water by mistake. Our **Random Forest** misting model scores an
> **R² of 0.92** on held-out data, predicting misting duration to within **1.8 seconds** on average.
> The feature importance confirms the biology too — **humidity is the strongest driver**, just as the
> oyster-mushroom literature predicts. These aren't mock-ups; they're measured results."

### 5:45 – 6:45 · Impact & Commercialization
[On screen: SDG 9 & 12 badges, cost figure, "add-on" framing]

> "The impact is direct: by ending over-watering, we project a **15 to 25 percent** improvement in usable
> yield, while cutting water waste — that's **UN SDG 12**, responsible production, and **SDG 9**, bringing
> innovation to agriculture. And it's **deployable today**. The whole rig costs around **600 Singapore
> dollars**, runs on a single Raspberry Pi, and installs as a **retrofit add-on** to a farm's existing
> shelves — no rebuild required. It's low-cost, low-risk, and it pays for itself in saved harvest."

### 6:45 – 7:15 · Future & Close
[On screen: future-potential icons, then team card]

> "Next, we're adding a **plain-language chatbot** so farmers can simply ask *'why is it watering?'*,
> a **CO₂ ventilation loop**, and support for **other crops**. SpotShrooms turns watching every mushroom —
> anytime, anywhere — from an impossible job into an automatic one. Thank you from **Team SpotShrooms,
> Singapore Polytechnic**."

---

## Production checklist
- [ ] Landscape (16:9), 1080p, clear audio (record narration separately if needed)
- [ ] **Names + "Singapore Polytechnic" in the first 30 seconds** (mandatory)
- [ ] Real **screen recording** of the dashboard (strongest 2 minutes — don't skip the live demo)
- [ ] Show the **prototype + wiring** on camera
- [ ] Total runtime **6:00–8:00**
- [ ] Upload to **YouTube as Public**, paste the link at final registration (by **30 June 2026**)
