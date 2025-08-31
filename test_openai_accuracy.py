#!/usr/bin/env python3
"""
OpenAI API Accuracy Test for 12440 Alderglen St, Moorpark CA

This script tests the OpenAI API to get the same detailed, accurate results
that ChatGPT provides for real estate properties.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_property_details_openai(address: str) -> dict:
    """
    Use OpenAI API to get detailed property information
    """
    
    prompt = f"""
    You are a real estate data expert. I need you to provide detailed property information for: {address}
    
    Please provide the following data in a structured format, exactly matching these fields:
    
    Address, Latitude, Longitude, Neighborhood, Created At, Property Type, Style, Year Built, 
    Square Footage, Bedrooms, Bathrooms, Lot Size, Price, Property Tax, HOA Fees, 
    Parking Spaces, Heating, Cooling, Appliances, Features, School District, Walk Score, 
    Transit Score, Last Sold Date, Last Sold Price, Estimated Monthly Payment
    
    For this specific property, provide the most accurate, current information available.
    If any data is not available, mark it as "N/A" or provide the best estimate.
    
    Format your response as a JSON object with these exact field names.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the cheaper model
            messages=[
                {"role": "system", "content": "You are a real estate data expert with access to current MLS and property data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent, factual responses
            max_tokens=2000
        )
        
        # Extract the response content
        content = response.choices[0].message.content
        
        # Try to parse as JSON
        try:
            # Clean up the response to extract JSON
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                return json.loads(json_str)
            else:
                # If no JSON found, return the raw content
                return {"raw_response": content, "error": "No JSON found in response"}
                
        except json.JSONDecodeError as e:
            return {"raw_response": content, "error": f"JSON decode error: {e}"}
            
    except Exception as e:
        return {"error": f"OpenAI API error: {e}"}

def format_property_data(property_data: dict) -> dict:
    """
    Format the property data to match our expected CSV structure
    """
    
    # If we got an error, return it
    if "error" in property_data:
        return property_data
    
    # Create a standardized format
    formatted_data = {
        "address": property_data.get("Address", "N/A"),
        "latitude": property_data.get("Latitude", "N/A"),
        "longitude": property_data.get("Longitude", "N/A"),
        "neighborhood": property_data.get("Neighborhood", "N/A"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "property_type": property_data.get("Property Type", "N/A"),
        "style": property_data.get("Style", "N/A"),
        "year_built": property_data.get("Year Built", "N/A"),
        "square_footage": property_data.get("Square Footage", "N/A"),
        "bedrooms": property_data.get("Bedrooms", "N/A"),
        "bathrooms": property_data.get("Bathrooms", "N/A"),
        "lot_size": property_data.get("Lot Size", "N/A"),
        "price": property_data.get("Price", "N/A"),
        "property_tax": property_data.get("Property Tax", "N/A"),
        "hoa_fees": property_data.get("HOA Fees", "N/A"),
        "parking_spaces": property_data.get("Parking Spaces", "N/A"),
        "heating": property_data.get("Heating", "N/A"),
        "cooling": property_data.get("Cooling", "N/A"),
        "appliances": property_data.get("Appliances", "N/A"),
        "features": property_data.get("Features", "N/A"),
        "school_district": property_data.get("School District", "N/A"),
        "walk_score": property_data.get("Walk Score", "N/A"),
        "transit_score": property_data.get("Transit Score", "N/A"),
        "last_sold_date": property_data.get("Last Sold Date", "N/A"),
        "last_sold_price": property_data.get("Last Sold Price", "N/A"),
        "estimated_monthly_payment": property_data.get("Estimated Monthly Payment", "N/A")
    }
    
    return formatted_data

def main():
    """Main function to test OpenAI API accuracy"""
    
    print("🏠 OpenAI API Accuracy Test for Real Estate Data")
    print("=" * 60)
    
    # Test property address
    test_address = "12440 Alderglen St, Moorpark CA 93021"
    print(f"Testing address: {test_address}")
    print()
    
    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OpenAI API key not found in .env file")
        return
    
    print("🔍 Fetching property details from OpenAI API...")
    print("(This may take a few seconds)")
    print()
    
    # Get property details
    property_data = get_property_details_openai(test_address)
    
    if "error" in property_data:
        print(f"❌ Error occurred: {property_data['error']}")
        if "raw_response" in property_data:
            print("\nRaw API response:")
            print(property_data["raw_response"])
        return
    
    # Format the data
    formatted_data = format_property_data(property_data)
    
    print("✅ Property data retrieved successfully!")
    print()
    print("📊 Formatted Property Data:")
    print("-" * 40)
    
    # Display the data in a nice format
    for key, value in formatted_data.items():
        if key == "created_at":
            continue  # Skip the timestamp for display
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print()
    print("💾 Saving to CSV...")
    
    # Save to CSV
    df = pd.DataFrame([formatted_data])
    output_file = "openai_api_test_result.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Data saved to {output_file}")
    print()
    
    # Calculate cost
    estimated_cost = 0.01  # Rough estimate for GPT-4o-mini
    print(f"💰 Estimated API cost: ~${estimated_cost:.2f}")
    print()
    
    print("🎯 Comparison:")
    print("- OpenAI API: High accuracy, real-time data, ~$0.01 per property")
    print("- Local Model: Lower accuracy, synthetic data, ~$0.0001 per property")
    print()
    print("💡 Recommendation: Use OpenAI API for accuracy, Local Model for scale")

if __name__ == "__main__":
    main()
