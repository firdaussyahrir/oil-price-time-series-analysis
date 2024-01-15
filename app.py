import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Function to read and preprocess the dataset
def read_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    cleaned = ['Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9']
    df = df.drop(columns=cleaned)
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Function to visualize cleaning results
def visualize_cleaning_results(df):
    st.subheader('Dataset yang digunakan')
    st.write(df.head(10))

# Function to visualize Closing Price graph
def visualize_closing_price(df):
    st.subheader('Visualisasi Harga Penutupan')
    st.line_chart(df['Close'])

# Function to visualize Moving Averages graph
def visualize_moving_averages(df, window_size=7):
    df['MA'] = df['Close'].rolling(window=window_size).mean().bfill()
    st.subheader(f'Visualisasi Moving Averages ({window_size} Hari)')
    st.line_chart(df[['Close', 'MA']])

# Function to visualize Exponential Smoothing prediction results
def visualize_exponential_smoothing_prediction(df, predict_series_es):
    st.subheader('Hasil Prediksi Exponential Smoothing ')
    
    # Visualization of Oil Prices and Exponential Smoothing Prediction
    combined_series = pd.concat([df['Close'], predict_series_es])
    st.line_chart(combined_series)

    # Display prediction series
    st.write(predict_series_es)


def main():
    st.title("Prediksi Harga Minyak dengan Analisis Time Series menggunakan Moving Average dan Exponential Smoothing")
    
    st.markdown("""
    Proyek ini bertujuan untuk melakukan prediksi harga minyak dengan memanfaatkan analisis time series menggunakan dua pendekatan utama: Moving Average (MA) dan Exponential Smoothing.

    Data yang digunakan mencakup:

    - Tanggal (Date)
    - Harga penutupan (Close)
    - Perubahan harga penutupan (chg(close))
    - Harga terendah (Low)
    - Perubahan harga terendah (chg(low))
    - Harga tertinggi (High)
    - Perubahan harga tertinggi (chg(high))
    """)

    st.markdown("Langkah-langkah proyek ini dimulai dengan pemahaman dan pembersihan data, diikuti oleh visualisasi tren harga minyak menggunakan Moving Average. Selanjutnya, penerapan model Exponential Smoothing untuk analisis time series dilakukan untuk memberikan prediksi yang lebih akurat. Hasil prediksi visualized menggunakan grafik, dan untuk kemudahan eksplorasi, aplikasi Streamlit telah dikembangkan.")

    st.markdown("Proyek ini memberikan wawasan tentang pergerakan harga minyak dengan memadukan kekuatan dari dua metode analisis time series yang berbeda, yaitu Moving Average untuk melihat tren jangka panjang dan Exponential Smoothing untuk merespons perubahan cepat dalam data.")
    
    # Read and preprocess data
    file_path = '/Users/firdaus./Desktop/MSIB @ Kalla Group/data/Copy of brentcrudeoil.xlsx'
    df = read_and_preprocess_data(file_path)

    # Visualize cleaning results
    visualize_cleaning_results(df)

    # Visualize Closing Price graph
    visualize_closing_price(df)

    # Visualize Moving Averages graph
    window_size = 7
    visualize_moving_averages(df, window_size)

    # Exponential Smoothing
    model_es = ExponentialSmoothing(df['Close'], trend='add', seasonal='add', seasonal_periods=7)
    fit_model_es = model_es.fit()
    predict_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
    predict_es = fit_model_es.predict(start=len(df), end=len(df) + len(predict_index) - 1)
    predict_series_es = pd.Series(predict_es.values, index=predict_index)

    # Visualize Exponential Smoothing prediction results
    visualize_exponential_smoothing_prediction(df, predict_series_es)

if __name__ == "__main__":
    main()
