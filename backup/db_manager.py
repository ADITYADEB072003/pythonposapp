import pymysql
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import sys

class DatabaseManager:
    """
    Manages the database connection, attempting to connect to MongoDB first,
    then falling back to a local MySQL instance on failure.
    """
    def __init__(self, mongo_uri, mysql_config):
        self.db_type = None
        self.client = None
        self.db = None
        self.conn = None
        self.cursor = None

        try:
            # 1. Attempt to connect to MongoDB (primary)
            print("Attempting to connect to primary database (MongoDB)...")
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster')
            self.db = self.client['pos_db']
            self.db_type = 'mongodb'
            print("✅ Successfully connected to MongoDB.")

        except ConnectionFailure as e:
            # 2. On failure, attempt to connect to MySQL (fallback)
            print(f"⚠️ MongoDB connection failed: {e}")
            print("Attempting to connect to fallback database (MySQL)...")
            try:
                self.conn = pymysql.connect(**mysql_config)
                self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)
                self.db_type = 'mysql'
                print("✅ Successfully connected to MySQL as a fallback.")
            except pymysql.MySQLError as mysql_err:
                print(f"❌ CRITICAL: Fallback MySQL connection also failed: {mysql_err}")
                # If both fail, there's nothing more to do.
                sys.exit("Both primary and fallback databases are unreachable. Exiting.")

    def get_db_type(self):
        """Returns the type of the current active database ('mongodb' or 'mysql')."""
        return self.db_type

    def get_inventory_collection(self):
        """Returns the MongoDB inventory collection if active."""
        if self.db_type == 'mongodb' and self.db:
            return self.db['inventory']
        return None

    def find_inventory_by_org(self, org_id):
        """
        Abstracted method to fetch inventory items. It uses the active
        database connection (MongoDB or MySQL).
        """
        if self.db_type == 'mongodb':
            # Using MongoDB
            inventory_col = self.db['inventory']
            return list(inventory_col.find({"org_id": org_id}).sort("name"))
        
        elif self.db_type == 'mysql':
            # Using MySQL
            self.cursor.execute("SELECT * FROM inventory WHERE org_id = %s ORDER BY name", (str(org_id),))
            return self.cursor.fetchall()
        
        return [] # Return empty list if no DB is active

    def close(self):
        """Closes the active database connection."""
        if self.db_type == 'mongodb' and self.client:
            self.client.close()
            print("MongoDB connection closed.")
        elif self.db_type == 'mysql' and self.conn:
            self.conn.close()
            print("MySQL connection closed.")
