from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
model = tf.keras.models.load_model('lstm_model.keras')

# Call model and make predictions
def make_predictions():
    data = pd.read_csv('data/data_daily.csv')
    data['# Date'] = pd.to_datetime(data['# Date'], format='%Y-%m-%d')
    data.set_index('# Date', inplace=True)
    monthly_data = data.resample('M').sum()
    receipt_counts = monthly_data['Receipt_Count']

    min_val = receipt_counts.min()
    max_val = receipt_counts.max()
    receipt_counts_scaled = (receipt_counts - min_val) / (max_val - min_val)

    seq_length = 3
    last_sequence = receipt_counts_scaled[-seq_length:].values.tolist()

    predictions = []
    for _ in range(12):
        X_pred = np.array(last_sequence[-seq_length:]).reshape(1, seq_length, 1)
        next_pred = model.predict(X_pred)
        next_pred_value = next_pred[0, 0]
        predictions.append(next_pred_value)
        last_sequence.append(next_pred_value)

    predictions = np.array(predictions)
    predictions = predictions * (max_val - min_val) + min_val

    dates_2022 = pd.date_range(start='2022-01-31', periods=12, freq='M')
    predictions_df = pd.DataFrame({'Date': dates_2022, 'Predicted Receipt Count': predictions})

    # Set 1-based indexing
    predictions_df.index = predictions_df.index + 1

    return predictions_df

# Plot predictions
def plot_predictions(predictions_df):
    data = pd.read_csv('data/data_daily.csv')
    data['# Date'] = pd.to_datetime(data['# Date'], format='%Y-%m-%d')
    data.set_index('# Date', inplace=True)
    monthly_data = data.resample('M').sum()

    plt.figure(figsize=(10, 6))
    plt.plot(monthly_data.index, monthly_data['Receipt_Count'], label='Actual')
    plt.plot(predictions_df['Date'], predictions_df['Predicted Receipt Count'], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Receipt Count')
    plt.title('Predicted vs Actual Monthly Receipt Counts')
    plt.legend()
    plt.tight_layout()

# Home Page
@app.route('/')
def index():
    predictions_df = make_predictions()
    plot_predictions(predictions_df)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Format numbers to avoid scientific notation and include commas for display
    predictions_df_display = predictions_df.copy()
    predictions_df_display['Predicted Receipt Count'] = predictions_df_display['Predicted Receipt Count'].apply(lambda x: f"{x:,.0f}")

    return render_template('index.html', plot_url='plot.png', tables=[predictions_df_display.to_html(classes='data')])

# Creates and serves the plot
@app.route('/plot.png')
def plot_png():
    predictions_df = make_predictions()
    plot_predictions(predictions_df)

    img = io.BytesIO()
    FigureCanvas(plt.gcf()).print_png(img)
    plt.close()
    return send_file(io.BytesIO(img.getvalue()), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)