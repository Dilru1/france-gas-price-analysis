import os
import pytest
from src.preprocessing import load_and_clean_data

def test_load_and_clean_data_logic(spark, tmp_path):
    """
    Tests if data is correctly loaded, dates are parsed, and coordinates fixed.
    """
    # 1. Setup: Create a dummy CSV file representing raw data
    # Raw format: id;cp;pop;lat;long;date;type;name;prix
    # Note: 4800000 lat should become 48.0
    csv_content = """1000001;75000;R;4800000;200000;2023-01-01T12:00:00;1;Gazole;1.999
1000002;69000;R;4500000;500000;2023-01-02T12:00:00;2;E10;1.888"""
    
    # Create a subfolder to match the function's expectation
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Write the file with the expected pattern "Prix*.csv.gz" (Spark reads .gz as text if not actually compressed)
    # We name it .csv.gz but write plain text for simplicity in testing (Spark handles this automatically)
    fake_file = data_dir / "Prix2023_test.csv.gz"
    fake_file.write_text(csv_content, encoding="utf-8")
    
    # 2. Execute: Call your function
    # We pass an empty config dict {} since your current code snippet doesn't use it in the active part
    df_result = load_and_clean_data(spark, str(data_dir), config={})

    # 3. Assert: Verify the results
    
    # Check Count
    assert df_result.count() == 2, "Should load 2 rows"
    
    # Check Columns exist
    expected_columns = ["id", "lat", "long", "date_parsed", "year", "month", "week", "prix"]
    for col in expected_columns:
        assert col in df_result.columns, f"Column {col} is missing"

    # Check Data Transformations
    row1 = df_result.filter("id = '1000001'").first()
    
    # Coordinate Check: 4800000 -> 48.0
    assert row1["lat"] == 48.0, f"Latitude should be divided by 100000, got {row1['lat']}"
    
    # Date Check
    assert row1["year"] == 2023, "Year extraction failed"
    assert row1["month"] == 1, "Month extraction failed"
    
    # Price Check (Cast to double)
    assert row1["prix"] == 1.999, "Price casting failed"