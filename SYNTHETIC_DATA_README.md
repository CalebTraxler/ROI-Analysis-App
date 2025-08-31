# Synthetic MLS Data Generator for San Francisco Properties

This tool generates synthetic MLS-like data for properties in San Francisco using GPT-4o-mini (cost-optimized) to fill in missing property details. It's designed to work with your existing ROI analysis system while providing realistic property data for ML model training.

## 🎯 Purpose

- **Fill Data Gaps**: Generate realistic MLS data for properties that only have coordinates
- **Cost Optimization**: Use GPT-4o-mini instead of more expensive models
- **San Francisco Focus**: Specifically designed for SF properties and neighborhoods
- **ML Model Training**: Provide synthetic data for real estate investment ML models

## 🏗️ Architecture

```
Coordinates → Reverse Geocoding → Address → GPT-4o-mini → Synthetic MLS Data → Database → CSV Export
```

## 📋 Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key** in `.env` file
3. **Internet Connection** for geocoding and API calls

## 🚀 Installation

1. **Install Dependencies**:
   ```bash
   pip install -r synthetic_requirements.txt
   ```

2. **Set Environment Variables**:
   Create a `.env` file in your project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🧪 Testing

### Step 1: Test Coordinate-to-Address Translation

Before running the full synthetic data generator, test the coordinate translation:

```bash
python test_coordinate_to_address.py
```

This will test:
- Reverse geocoding for SF coordinates
- Address extraction accuracy
- Neighborhood identification
- Rate limiting compliance

### Step 2: Test Synthetic Data Generation

Run a small test with the main generator:

```bash
python synthetic_data_generator.py
```

This processes 2 sample properties to verify everything works.

## 🔧 Usage

### Basic Usage

```python
from synthetic_data_generator import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator()

# Process properties (your existing property data)
properties_data = [
    {
        'osm_id': '12345',
        'latitude': 37.7749,
        'longitude': -122.4194,
        'building_type': 'house'
    }
    # ... more properties
]

# Generate synthetic data
generator.process_sf_properties(properties_data, max_properties=100)

# Export to CSV
df = generator.export_to_csv("synthetic_mls_data_sf.csv")

# Get statistics
stats = generator.get_processing_stats()
print(stats)
```

### Integration with Existing ROI System

```python
from openstreetmap_properties import OpenStreetMapProperties
from synthetic_data_generator import SyntheticDataGenerator

# Get existing property data
osm_fetcher = OpenStreetMapProperties()
properties = osm_fetcher.get_city_properties("San Francisco", "San Francisco", "CA")

# Generate synthetic data
generator = SyntheticDataGenerator()
generator.process_sf_properties(properties, max_properties=500)

# Export for ML model training
ml_training_data = generator.export_to_csv("ml_training_data_sf.csv")
```

## 📊 Generated Data Fields

The synthetic data includes realistic MLS fields:

- **Basic Info**: Property type, style, year built, square footage
- **Details**: Bedrooms, bathrooms, lot size, parking
- **Financial**: Price, property tax, HOA fees, estimated monthly payment
- **Features**: Appliances, notable features, school district
- **Scores**: Walk score, transit score
- **History**: Last sold date and price

## 🏘️ San Francisco Neighborhoods

The system recognizes and generates appropriate data for:

- **Mission District**: Victorian style, $1.2M avg
- **Outer Sunset**: Mid-century style, $1.4M avg
- **South of Market**: Modern style, $1.8M avg
- **Pacific Heights**: Classic style, $3.5M avg
- **Marina**: Mediterranean style, $2.2M avg
- **And more...**

## 💰 Cost Optimization

- **Model**: GPT-4o-mini (cheaper than GPT-4)
- **Rate Limiting**: 50 requests/minute
- **Batch Processing**: Process properties in controlled batches
- **Fallback Data**: Generates realistic data when API fails
- **Caching**: Avoids reprocessing existing properties

## 📈 Performance & Scaling

### Current Limits
- **Rate Limit**: 50 requests/minute
- **Processing Speed**: ~1.2 seconds per property
- **Daily Capacity**: ~72,000 properties (with 24/7 operation)

### Scaling Options
- **Parallel Processing**: Multiple API keys
- **Batch Optimization**: Process during off-peak hours
- **Selective Processing**: Focus on high-value neighborhoods

## 🔍 Monitoring & Logging

### Log Files
- `synthetic_data_generation.log`: Main processing logs
- `synthetic_mls_data.db`: SQLite database with all data

### Statistics
```python
stats = generator.get_processing_stats()
print(f"Total properties: {stats['total_properties']}")
print(f"By neighborhood: {stats['neighborhood_counts']}")
print(f"Recent activity: {stats['recent_properties']}")
```

## 🚨 Error Handling

The system includes robust error handling:

- **Geocoding Failures**: Skips properties with coordinate issues
- **API Failures**: Generates fallback data using neighborhood patterns
- **Rate Limiting**: Automatic delays and retry logic
- **Data Validation**: JSON parsing and fallback generation

## 📁 Output Files

1. **Database**: `synthetic_mls_data.db` (SQLite)
2. **CSV Export**: `synthetic_mls_data_sf.csv`
3. **Logs**: `synthetic_data_generation.log`

## 🔄 Workflow

### Phase 1: Testing (Recommended)
1. Run coordinate test script
2. Test with 2-5 sample properties
3. Verify data quality and format

### Phase 2: Small Batch
1. Process 50-100 properties
2. Review generated data quality
3. Adjust prompts if needed

### Phase 3: Full Scale
1. Process all SF properties
2. Monitor costs and performance
3. Export for ML model training

## 💡 Tips for Best Results

1. **Start Small**: Test with 5-10 properties first
2. **Monitor Costs**: Check OpenAI usage dashboard
3. **Validate Data**: Review generated data for realism
4. **Adjust Prompts**: Modify prompts based on output quality
5. **Use Off-Peak**: Process during low-usage hours

## 🚫 Limitations

- **API Dependencies**: Requires OpenAI API access
- **Rate Limits**: 50 requests/minute maximum
- **Coordinate Accuracy**: Depends on reverse geocoding quality
- **Data Realism**: Synthetic data may not match real MLS perfectly

## 🔮 Future Enhancements

- **Multiple API Keys**: Parallel processing
- **GIS Integration**: Precise neighborhood boundaries
- **Data Validation**: ML-based quality checking
- **Real-time Updates**: Continuous data generation
- **Multi-city Support**: Expand beyond San Francisco

## 📞 Support

For issues or questions:
1. Check the logs in `synthetic_data_generation.log`
2. Verify your OpenAI API key and credits
3. Test coordinate translation first
4. Start with small batches

## 🎉 Success Metrics

- **Coordinate Translation**: >90% success rate
- **Data Generation**: >95% success rate
- **Data Quality**: Realistic SF market values
- **Cost Efficiency**: <$0.01 per property

---

**Ready to generate synthetic MLS data for your San Francisco properties? Start with the test script and work your way up to full-scale processing!** 🚀
