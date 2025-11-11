# Why Baseline Data is Needed

## The Key Difference: Real-Time vs. Historical Patterns

### What TomTom API Provides:
- ✅ **Current traffic conditions** (right now, at this moment)
- ✅ Real-time speed data
- ✅ Current confidence levels

### What TomTom API Does NOT Provide:
- ❌ Historical patterns (how traffic behaves at 5 PM vs 10 AM)
- ❌ Weekly patterns (how Monday morning differs from Friday evening)
- ❌ Seasonal trends
- ❌ Past data needed to train ML models

## Why We Need Historical Data

### 1. **Machine Learning Models Need Training Data**

ML models learn patterns from historical examples. They need to see:
- How traffic behaves at different times of day
- How rush hours differ from off-peak hours
- How weekends differ from weekdays
- How traffic builds up over time

**Example:** The model learns that "Connaught Place at 5 PM on weekdays" typically has speed ~35 km/h, but at 10 AM it's ~55 km/h.

### 2. **Feature Engineering Requires Historical Data**

Look at the features we create in `ml/feature_engineering.py`:

```python
# These features NEED historical data:
"avg_speed_same_hour"          # Average speed at this hour (from past data)
"avg_speed_same_dow_hour"      # Average speed on this day+hour (from past weeks)
"avg_speed_7d"                 # 7-day rolling average
"speed_15min_ago"              # Speed 15 minutes ago (from stored data)
"speed_30min_ago"              # Speed 30 minutes ago
"speed_1h_ago"                 # Speed 1 hour ago
```

**Without historical data, these features would be empty or default values!**

### 3. **Prediction Requires Context**

When predicting traffic, the model considers:
- **Current state**: What TomTom API tells us right now
- **Historical context**: What normally happens at this time/location
- **Recent trends**: How traffic changed in the last hour

**Example:**
- TomTom says: "Current speed: 40 km/h"
- Historical data says: "At 5 PM on weekdays, this location averages 35 km/h"
- Model predicts: "Traffic is slightly better than usual, but will likely slow down (predicted: 32 km/h in 30 min)"

## The Data Collection Workflow

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Data Collection (1-2 weeks)                   │
│                                                          │
│  Every 15 minutes:                                      │
│  TomTom API → Fetch current data → Store in database    │
│                                                          │
│  Result: Thousands of data points with timestamps       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Feature Engineering                            │
│                                                          │
│  Load historical data from database                     │
│  Calculate:                                             │
│  - avg_speed_same_hour (from past data)                 │
│  - avg_speed_last_week                                  │
│  - speed_15min_ago, speed_30min_ago                     │
│                                                          │
│  Result: Rich feature set for each timestamp            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Model Training                                │
│                                                          │
│  Train LightGBM model on:                               │
│  - Features (historical patterns)                       │
│  - Targets (actual speeds from TomTom API)              │
│                                                          │
│  Model learns: "When it's 5 PM on Monday,                │
│                 and avg_speed_same_hour is 35 km/h,      │
│                 predict speed = 33 km/h"                 │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Prediction (Now Running)                       │
│                                                          │
│  When user requests prediction:                          │
│  1. Fetch current data from TomTom API                  │
│  2. Look up historical patterns from database           │
│  3. Calculate features (using historical data)         │
│  4. Model predicts based on learned patterns            │
│                                                          │
│  Result: Intelligent prediction, not just current state │
└─────────────────────────────────────────────────────────┘
```

## Can We Skip Baseline Data?

### Option 1: Use Only Current TomTom Data ❌
**Problem:** 
- Model can't learn patterns
- No historical features
- Predictions would just be "current speed ± small random variation"
- No better than just returning TomTom's current value

### Option 2: Use Pre-trained Models ❌
**Problem:**
- Models trained on different cities won't work well
- Traffic patterns are location-specific
- Need local data for accurate predictions

### Option 3: Start Collecting and Use Simple Heuristics Initially ✅
**Approach:**
1. Start collecting data immediately
2. Use simple rules while collecting (e.g., "rush hour = slower")
3. Once you have 1-2 weeks of data, train proper ML model
4. Switch to ML predictions

## Practical Example

**Scenario:** Predicting traffic at Connaught Place, Delhi

**Without Historical Data:**
```
User: "What's traffic like at 5 PM?"
System: "TomTom says 40 km/h right now"
→ Not very useful for planning ahead
```

**With Historical Data:**
```
User: "What's traffic like at 5 PM?"
System: 
  - Current: 40 km/h (from TomTom)
  - Historical average at 5 PM: 35 km/h
  - Predicted in 30 min: 32 km/h (traffic worsening)
  - Confidence: 85%
→ Much more useful for planning!
```

## Minimum Data Requirements

- **Minimum for training:** ~100 samples (can start with 1-2 days)
- **Recommended:** 1-2 weeks of data (captures weekly patterns)
- **Optimal:** 1+ month (captures monthly trends, anomalies)

## Summary

**You DO fetch from TomTom API**, but:
1. **Store it** in your database over time
2. **Build historical features** from stored data
3. **Train models** on patterns learned from history
4. **Combine** current TomTom data + historical patterns for intelligent predictions

The baseline data is simply **accumulated TomTom API responses stored over time** - it's not a separate data source!

