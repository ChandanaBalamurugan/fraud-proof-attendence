"""
Flask Web Dashboard for Fraud-Proof Attendance System
Displays attendance records with search, filter, charts, and export to PDF/Excel
"""

import os
import csv
import json
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from io import BytesIO, StringIO

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'),
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'))
CORS(app)

# Paths to attendance files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ATTENDANCE_FILE = os.path.join(BASE_DIR, "data", "attendance.csv")
ROOT_ATTENDANCE_FILE = os.path.join(PROJECT_ROOT, "data", "attendence.csv")

# Normalize paths
ATTENDANCE_FILE = os.path.normpath(ATTENDANCE_FILE)
ROOT_ATTENDANCE_FILE = os.path.normpath(ROOT_ATTENDANCE_FILE)


def get_attendance_data():
    """Read attendance data from CSV file"""
    file_to_read = ATTENDANCE_FILE if os.path.exists(ATTENDANCE_FILE) else ROOT_ATTENDANCE_FILE
    
    if not os.path.exists(file_to_read):
        return []
    
    records = []
    try:
        with open(file_to_read, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for row in reader:
                    if row.get('Name') and row.get('Date') and row.get('Time'):
                        records.append({
                            'name': row['Name'].strip(),
                            'date': row['Date'].strip(),
                            'time': row['Time'].strip(),
                            'datetime': f"{row['Date'].strip()} {row['Time'].strip()}"
                        })
    except Exception as e:
        print(f"Error reading attendance file: {e}")
    
    return records


@app.route('/')
def dashboard():
    """Render main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/attendance', methods=['GET'])
def attendance_api():
    """Get all attendance records with optional filtering"""
    records = get_attendance_data()
    
    # Get query parameters for filtering
    name_filter = request.args.get('name', '').strip().lower()
    from_date = request.args.get('from_date', '').strip()
    to_date = request.args.get('to_date', '').strip()
    
    # Apply filters
    filtered_records = records
    
    if name_filter:
        filtered_records = [r for r in filtered_records if name_filter in r['name'].lower()]
    
    if from_date:
        filtered_records = [r for r in filtered_records if r['date'] >= from_date]
    
    if to_date:
        filtered_records = [r for r in filtered_records if r['date'] <= to_date]
    
    # Sort by datetime descending (newest first)
    filtered_records.sort(key=lambda x: x['datetime'], reverse=True)
    
    return jsonify({
        'total': len(filtered_records),
        'records': filtered_records
    })


@app.route('/api/statistics', methods=['GET'])
def statistics_api():
    """Get attendance statistics"""
    records = get_attendance_data()
    
    if not records:
        return jsonify({
            'total_attendance': 0,
            'total_unique_people': 0,
            'attendance_today': 0,
            'attendance_by_person': {},
            'attendance_by_date': {}
        })
    
    # Count attendance by person
    attendance_by_person = defaultdict(int)
    attendance_dates_by_person = defaultdict(set)
    attendance_by_date = defaultdict(int)
    
    today = datetime.now().strftime("%Y-%m-%d")
    attendance_today = 0
    
    for record in records:
        person = record['name']
        date = record['date']
        
        attendance_by_person[person] += 1
        attendance_dates_by_person[person].add(date)
        attendance_by_date[date] += 1
        
        if date == today:
            attendance_today += 1
    
    # Sort by attendance count (descending)
    sorted_attendance = dict(sorted(attendance_by_person.items(), key=lambda x: x[1], reverse=True))
    
    return jsonify({
        'total_attendance': len(records),
        'total_unique_people': len(attendance_by_person),
        'attendance_today': attendance_today,
        'attendance_by_person': sorted_attendance,
        'attendance_by_date': dict(sorted(attendance_by_date.items(), reverse=True)[:30])  # Last 30 days
    })


@app.route('/api/export/excel', methods=['GET'])
def export_excel():
    """Export attendance records to Excel"""
    records = get_attendance_data()
    
    if not records:
        return jsonify({'error': 'No attendance records to export'}), 400
    
    # Create DataFrame
    df = pd.DataFrame(records)[['name', 'date', 'time']]
    df.columns = ['Name', 'Date', 'Time']
    
    # Sort by date descending
    df = df.sort_values('Date', ascending=False)
    
    # Write to BytesIO object
    output = BytesIO()
    df.to_excel(output, index=False, sheet_name='Attendance')
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'attendance_{datetime.now().strftime("%Y-%m-%d")}.xlsx'
    )


@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    """Export attendance records to CSV"""
    records = get_attendance_data()
    
    if not records:
        return jsonify({'error': 'No attendance records to export'}), 400
    
    # Create CSV content
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Date', 'Time'])
    
    for record in sorted(records, key=lambda x: x['datetime'], reverse=True):
        writer.writerow([record['name'], record['date'], record['time']])
    
    # Convert to bytes
    csv_bytes = output.getvalue().encode('utf-8')
    
    return send_file(
        BytesIO(csv_bytes),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attendance_{datetime.now().strftime("%Y-%m-%d")}.csv'
    )


@app.route('/api/people', methods=['GET'])
def people_list():
    """Get list of all people with attendance count"""
    records = get_attendance_data()
    
    people = defaultdict(int)
    for record in records:
        people[record['name']] += 1
    
    # Sort by name
    sorted_people = dict(sorted(people.items()))
    
    return jsonify({
        'people': [{'name': name, 'count': count} for name, count in sorted_people.items()]
    })


if __name__ == '__main__':
    print(f"Starting Attendance Dashboard Server...")
    print(f"Navigate to: http://localhost:5000")
    print(f"Press Ctrl+C to stop")
    app.run(debug=True, host='0.0.0.0', port=5000)
