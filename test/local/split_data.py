import csv
import math

# Function to write the split file
def create_split_file(split_rows, header, split_file_name):
	with open(split_file_name, 'w') as split_file:
		output = csv.writer(split_file, delimiter=",")

		# Add header to rows if you want
		split_rows.insert(0, header)
		output.writerows(split_rows)

# 
# 
# CHANGE THESE TO MATCH YOUR SPECIFICATIONS
# 
# 

split_file_rows = 40755
file_path = "data.csv"


# Read and split the rows
split_rows = []
with open(file_path) as csv_file:
	# Read the CSV file
	csv_reader = csv.reader(csv_file, delimiter=',')
	# Loop over all rows to get the number of row and all values of the row
	for row_num, row in enumerate(csv_reader):
		# Get the header from the first row
		if row_num == 0:
			header = row
		else:
			# If we have reached the split mark, create a split file
			if row_num % split_file_rows == 0:
				# Get the number of the split file and convert to string
				file_suffix = str(int(row_num / split_file_rows))
				# Create the name of the split file
				split_file_name = "split_file_" + file_suffix + ".csv"	

				# Output the split file
				create_split_file(split_rows, header, split_file_name)

				# Make split rows start with this row
				split_rows = [row]
			# If we haven't reached the split point, add the row to the current split_row
			else:
				split_rows.append(row)

	# When it is the end of file then output all the leftover rows
	split_rows.append(row)
	file_suffix = str(math.ceil(row_num / split_file_rows))
	split_file_name = "split_file_" + file_suffix + ".csv"	

	create_split_file(split_rows, header, split_file_name)
