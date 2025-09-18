import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'some_secret_key'

# MongoDB connection
client = MongoClient("mongodb+srv://adityadeb:eCunNWFwpyZpHdul@testid.hyqwjw5.mongodb.net/?retryWrites=true&w=majority&appName=testid")
db = client['pos_db']

# Collections
inventory_col = db['inventory']
sales_col = db['sales']
users_col = db['users']

# Upload config
UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_doc):
        self.id = str(user_doc['_id'])
        self.username = user_doc['username']
        self.password_hash = user_doc['password_hash']
        self.org_id = user_doc.get('org_id')

@login_manager.user_loader
def load_user(user_id):
    user_doc = users_col.find_one({"_id": ObjectId(user_id)})
    if user_doc:
        return User(user_doc)
    return None

@app.route('/register', methods=['GET', 'POST'])
@app.route('/register', methods=['GET', 'POST'])
# def register():
#     organizations = list(db['organizations'].find())
#     if request.method == 'POST':
#         username = request.form['username'].strip()
#         password = request.form['password']
#         org_id = request.form.get('org_id')

#         if not org_id:
#             flash("Organization selection is required.", "danger")
#             return redirect(url_for("register"))

#         if users_col.find_one({"username": username}):
#             flash("Username already exists", "danger")
#             return redirect(url_for("register"))

#         hashed_pw = generate_password_hash(password)
#         users_col.insert_one({
#             "username": username,
#             "password_hash": hashed_pw,
#             "org_id": ObjectId(org_id)
#         })
#         flash("Registration successful! Please log in.", "success")
#         return redirect(url_for("login"))

#     return render_template('register.html', organizations=organizations)

@app.route('/register', methods=['GET', 'POST'])
@app.route('/register', methods=['GET', 'POST'])
def register():
    organizations = list(db['organizations'].find())
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        org_id = request.form.get('org_id')
        new_org_name = request.form.get('new_org_name', '').strip()

        if not org_id:
            flash("Organization selection is required.", "danger")
            return redirect(url_for("register"))

        # Handle new organization creation with case-insensitive duplicate check
        if org_id == 'new':
            if not new_org_name:
                flash("New organization name is required.", "danger")
                return redirect(url_for("register"))
            # Case-insensitive search for existing organization
            existing_org = db['organizations'].find_one({
                "name": {"$regex": f"^{new_org_name}$", "$options": "i"}
            })
            if existing_org:
                org_id = existing_org['_id']
            else:
                org_res = db['organizations'].insert_one({"name": new_org_name})
                org_id = org_res.inserted_id
        else:
            org_id = ObjectId(org_id)

        if users_col.find_one({"username": username}):
            flash("Username already exists", "danger")
            return redirect(url_for("register"))

        hashed_pw = generate_password_hash(password)
        users_col.insert_one({
            "username": username,
            "password_hash": hashed_pw,
            "org_id": org_id
        })
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template('register.html', organizations=organizations)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        user_doc = users_col.find_one({"username": username})
        if user_doc and check_password_hash(user_doc['password_hash'], password):
            user = User(user_doc)
            login_user(user)
            flash(f"Logged in as {username}", "success")
            return redirect(url_for('inventory'))
        else:
            flash("Invalid username or password", "danger")

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

@app.route('/')
def home():
    return redirect(url_for('inventory'))

@app.route('/inventory', methods=['GET', 'POST'])
@login_required
def inventory():
    if request.method == 'POST':
        name = request.form['name'].strip()
        file = request.files.get('image')
        filename = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        try:
            price = float(request.form['price'])
            quantity = int(request.form['quantity'])
        except ValueError:
            flash("Invalid price or quantity", "danger")
            return redirect(url_for('inventory'))

        existing = inventory_col.find_one({"name": name, "org_id": ObjectId(current_user.org_id)})
        item_data = {
            "name": name,
            "price": price,
            "quantity": quantity,
            "org_id": ObjectId(current_user.org_id)
        }
        if filename:
            item_data["image"] = filename

        if existing:
            inventory_col.update_one({"_id": existing['_id']}, {"$set": item_data})
            flash(f"Updated {name}", "success")
        else:
            inventory_col.insert_one(item_data)
            flash(f"Added new item {name}", "success")

        return redirect(url_for('inventory'))

    items = list(inventory_col.find({"org_id": ObjectId(current_user.org_id)}).sort("name"))
    return render_template('inventory.html', items=items)

@app.route('/inventory/delete/<item_id>', methods=['POST'])
@login_required
def delete_inventory(item_id):
    item = inventory_col.find_one({"_id": ObjectId(item_id), "org_id": ObjectId(current_user.org_id)})
    if item:
        inventory_col.delete_one({"_id": ObjectId(item_id)})
        flash("Item deleted.", "info")
    else:
        flash("Item not found or permission denied.", "danger")
    return redirect(url_for('inventory'))

@app.route('/billing', methods=['GET', 'POST'])
@login_required
def billing():
    cart = session.get("cart", {})

    if request.method == "POST":
        item_id = request.form["item_id"]
        qty = int(request.form["quantity"])
        product = inventory_col.find_one({"_id": ObjectId(item_id), "org_id": ObjectId(current_user.org_id)})
        if not product:
            flash("Invalid product.", "danger")
            return redirect(url_for('billing'))
        cart[item_id] = cart.get(item_id, 0) + qty
        session["cart"] = cart
        flash("Added to cart.", "success")
        return redirect(url_for("billing"))

    search = request.args.get("search", "")
    query = {"org_id": ObjectId(current_user.org_id)}
    if search:
        query["name"] = {'$regex': search, '$options': 'i'}

    items = list(inventory_col.find(query))
    cart_items = []
    total = 0
    for item_id, qty in cart.items():
        product = inventory_col.find_one({"_id": ObjectId(item_id), "org_id": ObjectId(current_user.org_id)})
        if product:
            product["cart_qty"] = qty
            product["subtotal"] = qty * product["price"]
            product["_id"] = str(product["_id"])
            cart_items.append(product)
            total += product["subtotal"]

    return render_template("billing.html", items=items, cart=cart_items, total=total, search=search)

@app.route('/cart/delete/<item_id>', methods=['POST'])
@login_required
def delete_cart_item(item_id):
    cart = session.get("cart", {})
    if item_id in cart:
        cart.pop(item_id)
        session["cart"] = cart
        flash("Item removed from cart.", "info")
    return redirect(url_for("billing"))

@app.route('/checkout', methods=['POST'])
@login_required
def checkout():
    cart = session.get("cart", {})
    bill_items = []
    for item_id, qty in cart.items():
        product = inventory_col.find_one({"_id": ObjectId(item_id), "org_id": ObjectId(current_user.org_id)})
        if not product or product['quantity'] < qty:
            flash(f"Not enough stock for {product['name'] if product else 'an item'} (available: {product['quantity'] if product else 0}).", "danger")
            return redirect(url_for("billing"))

    for item_id, qty in cart.items():
        product = inventory_col.find_one({"_id": ObjectId(item_id), "org_id": ObjectId(current_user.org_id)})
        inventory_col.update_one({"_id": ObjectId(item_id)}, {"$inc": {"quantity": -qty}})
        sales_col.insert_one({
            "name": product["name"],
            "quantity": qty,
            "price": product['price'],
            "org_id": ObjectId(current_user.org_id),
            "date": datetime.utcnow()
        })
        bill_items.append({
            "name": product["name"],
            "quantity": qty,
            "price": product['price']
        })

    session["last_bill"] = bill_items
    session["cart"] = {}
    flash("Checkout successful!", "success")
    return redirect(url_for("bill"))

@app.route('/bill')
@login_required
def bill():
    cart = session.get("last_bill", [])
    total = sum(item['price'] * item['quantity'] for item in cart)
    return render_template('bill.html', cart=cart, total=total)

@app.route('/sales', methods=['GET', 'POST'])
@login_required
def sales_history():
    start = request.args.get('start')
    end = request.args.get('end')

    query = {'org_id': ObjectId(current_user.org_id)}
    if start and end:
        try:
            start_dt = datetime.strptime(start, '%Y-%m-%d')
            end_dt = datetime.strptime(end, '%Y-%m-%d')
            query['date'] = {'$gte': start_dt, '$lte': end_dt}
        except:
            flash("Invalid date format. Use YYYY-MM-DD", "danger")

    sales = list(sales_col.find(query).sort('date', 1))

    date_sales = {}
    for sale in sales:
        if not sale.get('date'):
            continue
        dt = sale['date'].strftime('%Y-%m-%d')
        date_sales[dt] = date_sales.get(dt, 0) + sale['price'] * sale['quantity']

    dates = list(date_sales.keys())
    sales_values = list(date_sales.values())

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(dates, sales_values)
    ax.set_title('Sales Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Sales (₹)')
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    chart_data = base64.b64encode(buf.getvalue()).decode()

    return render_template('sales.html', sales=sales, chart_data=chart_data, start=start, end=end)

@app.route('/sales_chart.png')
@login_required
def sales_chart():
    org_id = ObjectId(current_user.org_id)
    sales = list(sales_col.find({"org_id": org_id}).sort('date', 1))

    date_sales = {}
    for sale in sales:
        if not sale.get('date'):
            continue
        dt = sale['date'].strftime('%Y-%m-%d')
        date_sales[dt] = date_sales.get(dt, 0) + sale['price'] * sale['quantity']

    dates = list(date_sales.keys())
    sales_values = list(date_sales.values())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dates, sales_values)
    ax.set_title('Sales Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Sales (₹)')
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
