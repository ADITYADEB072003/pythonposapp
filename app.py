import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'some_secret_key'

# MongoDB connection
client = MongoClient("mongodb+srv://adityadeb:eCunNWFwpyZpHdul@testid.hyqwjw5.mongodb.net/?retryWrites=true&w=majority&appName=testid")
db = client['pos_db']
inventory_col = db['inventory']
sales_col = db['sales']

# Upload config
UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return redirect(url_for('inventory'))

@app.route('/inventory', methods=['GET', 'POST'])
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

        existing = inventory_col.find_one({"name": name})
        item_data = {
            "name": name,
            "price": price,
            "quantity": quantity
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

    items = list(inventory_col.find().sort("name"))
    return render_template('inventory.html', items=items)

@app.route('/inventory/delete/<item_id>')
def delete_inventory(item_id):
    inventory_col.delete_one({"_id": ObjectId(item_id)})
    flash("Item deleted.", "info")
    return redirect(url_for('inventory'))

@app.route('/billing', methods=['GET', 'POST'])
def billing():
    cart = session.get("cart", {})

    if request.method == "POST":
        item_id = request.form["item_id"]
        qty = int(request.form["quantity"])
        cart[item_id] = cart.get(item_id, 0) + qty
        session["cart"] = cart
        flash("Added to cart.", "success")
        return redirect(url_for("billing"))

    search = request.args.get("search", "")
    query = {}
    if search:
        query = {"name": {'$regex': search, '$options': 'i'}}

    items = list(inventory_col.find(query))
    cart_items = []
    total = 0
    for item_id, qty in cart.items():
        product = inventory_col.find_one({"_id": ObjectId(item_id)})
        if product:
            product["cart_qty"] = qty
            product["subtotal"] = qty * product["price"]
            cart_items.append(product)
            total += product["subtotal"]

    return render_template("billing.html", items=items, cart=cart_items, total=total, search=search)

@app.route('/checkout', methods=['POST'])
def checkout():
    cart = session.get("cart", {})
    bill_items = []
    for item_id, qty in cart.items():
        product = inventory_col.find_one({"_id": ObjectId(item_id)})
        if product and product['quantity'] >= qty:
            inventory_col.update_one({"_id": ObjectId(item_id)}, {"$inc": {"quantity": -qty}})
            sales_col.insert_one({
                "name": product["name"],
                "quantity": qty,
                "price": product['price'],
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
def bill():
    cart = session.get("last_bill", [])
    total = sum(item['price'] * item['quantity'] for item in cart)
    return render_template('bill.html', cart=cart, total=total)

@app.route('/sales', methods=['GET', 'POST'])
def sales_history():
    start = request.args.get('start')
    end = request.args.get('end')

    query = {}
    if start and end:
        try:
            start_dt = datetime.strptime(start, '%Y-%m-%d')
            end_dt = datetime.strptime(end, '%Y-%m-%d')
            query = {'date': {'$gte': start_dt, '$lte': end_dt}}
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
def sales_chart():
    # generate plot as before
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Response(buf.getvalue(), mimetype='image/png')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)


