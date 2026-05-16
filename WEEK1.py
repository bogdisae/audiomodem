import csv

# Vector (Python list) to store the values
vector = []

# Open the CSV file
with open("data.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        # Check row exists and first column is not empty
        if row and row[0].strip() != "":
            vector.append(row[0])

print(vector)