import pymongo

def fetch_data(collection):
    # Fetch all documents from the collection
    data = collection.find()
    return data

def generate_html(data):
    # Create the HTML structure for data_cand.html
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data from MongoDB</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                padding: 20px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h1>Data from MongoDB</h1>
        <table>
            <tr>
                <th>Name</th>
                <th>Phone</th>
                <th>Email</th>
                <th>Skills</th>
            </tr>
    '''

    # Populate the table with data
    for record in data:
        skills_list = ', '.join(record.get('skills', []))  # Join skills into a string
        html_content += f'''
            <tr>
                <td>{record.get('name', 'N/A')}</td>
                <td>{record.get('phone', 'N/A')}</td>
                <td>{record.get('email', 'N/A')}</td>
                <td>{skills_list}</td>
            </tr>
        '''

    html_content += '''
            </table>
        </body>
    </html>
    '''

    return html_content

if __name__ == "__main__":
    print("Hello mongo")
    
    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb+srv://khushalibegde18:y1ESj3rHlikmgxu6@cluster0.9q3rt.mongodb.net/")
    print(client)
    
    db = client['resume_analyzer']  # Make sure the database name is correct
    collection = db['candidates']     # Ensure the collection name is correct
    
    # Fetch data and generate HTML
    data = fetch_data(collection)
    data_list = list(data)  # Convert to list to reuse in HTML generation
    
    # Generate and write data_cand.html
    cand_html_content = generate_html(data_list)
    with open('data_cand.html', 'w') as file:
        file.write(cand_html_content)

    print("Data has been written to data_cand.html")
