"""
README for Urban-Geo Traffic Prediction System

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up PostgreSQL with PostGIS:**
   ```bash
   # Install PostGIS extension
   psql -U postgres -d urban_geo -c "CREATE EXTENSION IF NOT EXISTS postgis;"
   
   # Initialize schema
   python database/init_db.py
   ```

3. **Configure database:**
   - Update `config.py` with your database credentials
   - Ensure TomTom API key is set in `config.py`

4. **Add monitoring locations:**
   ```bash
   python utils/add_locations.py --default  # Add default locations
   # Or add custom location:
   python utils/add_locations.py --lat 28.6139 --lon 77.2090 --name "Connaught Place" --priority 2
   ```

## Data Collection

### Manual Collection
```bash
python data_collection/collect_data.py
```

### Scheduled Collection (Cron)
Add to crontab for every 15 minutes:
```bash
*/15 * * * * cd /path/to/urban-geo && python data_collection/collect_data.py >> logs/collection.log 2>&1
```

### Using Celery (Optional)
```bash
# Start Celery worker
celery -A data_collection.celery_app worker --loglevel=info

# Start Celery beat scheduler
celery -A data_collection.celery_app beat --loglevel=info
```

## Model Training

### Train Current Traffic Model
```bash
python ml/model_training.py
```

### Train Future Prediction Model
Edit `ml/model_training.py` to set `model_type="future_30min"` and run.

## API Server

### Start API Server
```bash
python run_server.py
# Or
uvicorn server.api:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- **GET /health** - Health check
- **POST /predict/current** - Predict current traffic
  ```json
  {
    "lat": 28.6139,
    "lon": 77.2090
  }
  ```
- **POST /predict/future** - Predict future traffic
  ```json
  {
    "lat": 28.6139,
    "lon": 77.2090,
    "minutes_ahead": 30
  }
  ```
- **GET /stats** - Get historical statistics

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## Project Structure

```
urban-geo/
├── config.py                 # Configuration (DB, API keys)
├── database/
│   ├── schema.sql           # Database schema
│   └── init_db.py          # Database initialization
├── data_collection/
│   ├── collector.py        # Data collection logic
│   └── collect_data.py     # Scheduled collection script
├── ml/
│   ├── feature_engineering.py  # Feature engineering
│   └── model_training.py       # Model training
├── server/
│   └── api.py              # FastAPI application
├── utils/
│   ├── helpers.py          # Utility functions
│   └── add_locations.py    # Location management
├── models/                 # Saved models (auto-created)
├── logs/                   # Log files (auto-created)
└── requirements.txt        # Python dependencies
```

## Development Workflow

1. **Phase 1: Data Collection**
   - Set up database schema
   - Add monitoring locations
   - Run data collection for 1-2 weeks

2. **Phase 2: Model Training**
   - Collect sufficient data (minimum 100 samples)
   - Train initial model
   - Evaluate performance

3. **Phase 3: API Deployment**
   - Start API server
   - Test endpoints
   - Monitor performance

4. **Phase 4: Future Prediction**
   - Extend model with time-series features
   - Train future prediction model
   - Deploy and test

## Monitoring

- Check API logs: `tail -f logs/api.log`
- Check collection logs: `tail -f logs/collection.log`
- Monitor database: Query `api_logs` table for API metrics
- Monitor model performance: Query `model_metadata` table

## Rate Limiting

TomTom API: 2,500 requests/day
- For 50 locations: Poll every ~30 minutes
- For 100 locations: Poll every ~60 minutes
- Adjust based on priority levels

## Troubleshooting

1. **Database connection errors:**
   - Check `config.py` credentials
   - Verify PostgreSQL is running
   - Ensure PostGIS extension is installed

2. **Model not found errors:**
   - Train models first: `python ml/model_training.py`
   - Check `models/` directory for saved models

3. **API prediction errors:**
   - Ensure sufficient historical data exists
   - Check model is loaded correctly
   - Verify feature engineering pipeline

## Production Considerations

1. **Environment Variables:**
   - Move sensitive config to environment variables
   - Use `.env` file (not committed to git)

2. **Model Registry:**
   - Implement model versioning system
   - Store model metadata in database
   - A/B test model versions

3. **Caching:**
   - Cache frequent predictions
   - Use Redis for distributed caching

4. **Monitoring:**
   - Set up Prometheus metrics
   - Configure alerting for failures
   - Monitor API latency and accuracy

5. **Scaling:**
   - Use load balancer for API servers
   - Scale horizontally for high traffic
   - Consider async processing for batch predictions

