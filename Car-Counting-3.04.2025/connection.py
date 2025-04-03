try:
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-KHKO40I\\SQLEXPRESS;"
        "DATABASE=Car-count;"
        "Trusted_Connection=yes;"
    )
    cursor = conn.cursor()
    print("Connected to SQL Server successfully!")

    # Ensure the table exists (create it if not)
    cursor.execute("""
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Count_Data')
    BEGIN
        CREATE TABLE Count_Data (
            id INT IDENTITY(1,1) PRIMARY KEY,
            Entry_Count INT,
            Exit_Count INT,
            Timestamps TIME(7),
            Date DATE
        )
    END
    """)
    conn.commit()
    print("Table checked/created successfully.")

except pyodbc.Error as e:
    print("Failed to connect to SQL Server.")
    print("Error:", e)
    exit()

def update_database(entry_count, exit_count):
    """Update the single record in Count_Data table with latest counts."""
    try:
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')

        # Check if there is an existing record
        cursor.execute("SELECT COUNT(*) FROM Count_Data")
       
        record_count = cursor.fetchone()[0]
        if record_count == 0:
            # If no record exists, insert a new one
            cursor.execute("""
                INSERT INTO Count_Data (Entry_Count, Exit_Count, Timestamps, Date)
                VALUES (?, ?, ?, ?)
            """, (entry_count, exit_count, current_time, current_date))
        else:
            # If record exists, update it
            cursor.execute("""
                UPDATE Count_Data
                SET Entry_Count = ?, Exit_Count = ?, Timestamps = ?, Date = ?
            """, (entry_count, exit_count, current_time, current_date))

        conn.commit()
        print("Database record updated successfully.")

    except pyodbc.Error as e:
        print("Database update failed:", e)
