# ... other imports
from backup.db_manager import DatabaseManager # <-- Import the new class

# --- Database Configurations ---
MONGO_URI = "mongodb+srv://adityadeb:eCunNWFwpyZpHdul@testid.hyqwjw5.mongodb.net/?retryWrites=true&w=majority&appName=testid"
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # <-- IMPORTANT: Fill in your MySQL root password
    'database': 'pos_db',
    'autocommit': True # Good for simple web apps
}

# --- Initialize the Database Manager ---
# This single line handles the connection and fallback logic!
db_manager = DatabaseManager(mongo_uri=MONGO_URI, mysql_config=MYSQL_CONFIG)


# --- Example: Modified Inventory Route ---
@app.route('/inventory', methods=['GET', 'POST'])
@login_required
def inventory():
    org_id = ObjectId(current_user.org_id)
    
    # You can check which database is active
    if db_manager.get_db_type() == 'mongodb':
        # Full functionality with MongoDB
        inventory_col = db_manager.get_inventory_collection()
        if request.method == 'POST':
            # Your complete POST logic for MongoDB...
            # Example:
            # item_code = request.form['item_code'].strip()
            # inventory_col.insert_one(...)
            pass
        
        items = list(inventory_col.find({"org_id": org_id}).sort("name"))
        return render_template('inventory.html', items=items, db_type='MongoDB')

    elif db_manager.get_db_type() == 'mysql':
        # Limited functionality with MySQL fallback
        # In fallback mode, you might disable write operations
        if request.method == 'POST':
            flash("Database is in fallback mode. Adding/editing items is disabled.", "warning")
            return redirect(url_for('inventory'))
            
        # Use the abstracted method to fetch data
        items = db_manager.find_inventory_by_org(org_id)
        return render_template('inventory.html', items=items, db_type='MySQL (Fallback)')
    
    # This case should not be reached if the app exits on total failure
    flash("No database connection available.", "danger")
    return render_template('inventory.html', items=[], db_type='None')

# You would continue to adapt other routes (billing, sales) in a similar fashion.
