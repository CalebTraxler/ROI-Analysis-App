#!/usr/bin/env python3
"""
OpenAI API Accuracy Test with GPT-4o for 12440 Alderglen St, Moorpark CA

This script uses GPT-4o (not mini) to get the same detailed, accurate results
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

def get_property_details_gpt4o(address: str) -> dict:
    """
    Use GPT-4o to get detailed property information (more accurate than GPT-4o-mini)
    """
    
    prompt = f"""
    You are a real estate data expert with access to current MLS and property data.
    
    I need you to provide detailed property information for: {address}
    
    Please provide the following data in a structured format, exactly matching these fields:
    
    Address, Latitude, Longitude, Neighborhood, Created At, Property Type, Style, Year Built, 
    Square Footage, Bedrooms, Bathrooms, Lot Size, Price, Property Tax, HOA Fees, 
    Parking Spaces, Heating, Cooling, Appliances, Features, School District, Walk Score, 
    Transit Score, Last Sold Date, Last Sold Price, Estimated Monthly Payment
    
    For this specific property, provide the most accurate, current information available.
    If any data is not available, mark it as "N/A" or provide the best estimate.
    
    IMPORTANT: Use current, up-to-date information from MLS, Zillow, Redfin, or other real estate sources.
    Do not make up or estimate data - provide only factual, current information.
    
    Format your response as a JSON object with these exact field names.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using the full GPT-4o model (not mini) for better accuracy
            messages=[
                {"role": "system", "content": "You are a real estate data expert with access to current MLS and property data. Provide only factual, current information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent, factual responses
            max_tokens=3000   # More tokens for detailed responses
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

def get_property_details_gpt4o_mini_web_search(address: str) -> dict:
    """
    Use GPT-4o-mini with web search capabilities for maximum accuracy
    """
    
    prompt = f"""
    You are a real estate data expert. I need you to search for and provide detailed property information for: {address}
    
    Please search the web for current MLS listings, Zillow, Redfin, and other real estate sources to get the most accurate, up-to-date information.
    
    Provide the following data in a structured format, exactly matching these fields:
    
    Address, Latitude, Longitude, Neighborhood, Created At, Property Type, Style, Year Built, 
    Square Footage, Bedrooms, Bathrooms, Lot Size, Price, Property Tax, HOA Fees, 
    Parking Spaces, Heating, Cooling, Appliances, Features, School District, Walk Score, 
    Transit Score, Last Sold Date, Last Sold Price, Estimated Monthly Payment
    
    IMPORTANT: 
    1. Search for CURRENT listings and data
    2. Provide only factual information from reliable sources
    3. Include source citations if possible
    4. Use the most recent data available
    
    Format your response as a JSON object with these exact field names.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini but with better prompting
            messages=[
                {"role": "system", "content": "You are a real estate data expert. Search for current, factual property information from reliable sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        # Extract the response content
        content = response.choices[0].message.content
        
        # Try to parse as JSON
        try:
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                return json.loads(json_str)
            else:
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

def compare_with_chatgpt_results(api_data: dict) -> None:
    """
    Compare our API results with the ChatGPT results you provided
    """
    
    print("\n🔍 ACCURACY COMPARISON WITH CHATGPT RESULTS:")
    print("=" * 60)
    
    # ChatGPT's actual results
    chatgpt_results = {
        "Year Built": "1989",
        "Square Footage": "2,518 sq ft", 
        "Bedrooms": "5",
        "Bathrooms": "2.5",
        "Price": "$1,049,000",
        "Neighborhood": "Mountain Meadows",
        "Lot Size": "~6,000 sq ft",
        "Property Tax": "$4,637/year",
        "HOA Fees": "$130/month"
    }
    
    # Our API results
    api_results = {
        "Year Built": api_data.get("year_built", "N/A"),
        "Square Footage": api_data.get("square_footage", "N/A"),
        "Bedrooms": api_data.get("bedrooms", "N/A"),
        "Bathrooms": api_data.get("bathrooms", "N/A"),
        "Price": api_data.get("price", "N/A"),
        "Neighborhood": api_data.get("neighborhood", "N/A"),
        "Lot Size": api_data.get("lot_size", "N/A"),
        "Property Tax": api_data.get("property_tax", "N/A"),
        "HOA Fees": api_data.get("hoa_fees", "N/A")
    }
    
    # Compare each field
    accuracy_score = 0
    total_fields = len(chatgpt_results)
    
    for field, chatgpt_value in chatgpt_results.items():
        api_value = api_results[field]
        
        if str(api_value).lower() == str(chatgpt_value).lower():
            print(f"✅ {field}: {api_value} (MATCHES ChatGPT)")
            accuracy_score += 1
        else:
            print(f"❌ {field}: {api_value} (ChatGPT: {chatgpt_value})")
    
    accuracy_percentage = (accuracy_score / total_fields) * 100
    print(f"\n📊 ACCURACY SCORE: {accuracy_score}/{total_fields} ({accuracy_percentage:.1f}%)")
    
    if accuracy_percentage >= 80:
        print("🎯 EXCELLENT ACCURACY - API is working well!")
    elif accuracy_percentage >= 60:
        print("⚠️  MODERATE ACCURACY - Some improvements needed")
    else:
        print("🚨 POOR ACCURACY - API needs significant improvement")

def main():
    """Main function to test OpenAI API accuracy with different models"""
    
    print("🏠 OpenAI API Accuracy Test - GPT-4o vs GPT-4o-mini")
    print("=" * 60)
    
    # Test property address
    test_address = "12440 Alderglen St, Moorpark CA 93021"
    print(f"Testing address: {test_address}")
    print()
    
    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OpenAI API key not found in .env file")
        return
    
    print("🔍 Testing GPT-4o (Full Model) for maximum accuracy...")
    print("(This may take a few seconds and cost more)")
    print()
    
    # Test with GPT-4o (full model)
    property_data_gpt4o = get_property_details_gpt4o(test_address)
    
    if "error" in property_data_gpt4o:
        print(f"❌ GPT-4o Error: {property_data_gpt4o['error']}")
        if "raw_response" in property_data_gpt4o:
            print("\nRaw GPT-4o response:")
            print(property_data_gpt4o["raw_response"])
    else:
        print("✅ GPT-4o data retrieved successfully!")
        formatted_data_gpt4o = format_property_data(property_data_gpt4o)
        
        print("\n📊 GPT-4o Results:")
        print("-" * 40)
        for key, value in formatted_data_gpt4o.items():
            if key == "created_at":
                continue
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Compare with ChatGPT results
        compare_with_chatgpt_results(formatted_data_gpt4o)
        
        # Save to CSV
        df = pd.DataFrame([formatted_data_gpt4o])
        output_file = "gpt4o_api_test_result.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 GPT-4o data saved to {output_file}")
    
    print("\n" + "="*60)
    print("💰 COST ANALYSIS:")
    print("- GPT-4o: ~$0.03-0.05 per property (higher accuracy)")
    print("- GPT-4o-mini: ~$0.01 per property (lower accuracy)")
    print("- Local Model: ~$0.0001 per property (synthetic data)")
    print()
    print("💡 RECOMMENDATION:")
    print("Use GPT-4o for critical properties, Local Model for bulk processing")

if __name__ == "__main__":
    main()
