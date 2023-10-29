## Project Description

**Project Goals and Features**

> Analyze the functional requirements of the stock price comprehensive analysis and prediction tool, and investigate its design and implementation techniques. Design the overall structure of the stock price comprehensive analysis and prediction tool, implementing the following features:
>
> - Display of major market indices, stock comparison analysis, and individual stock information analysis
>
> - Predictions for tomorrow's stock price, price range forecast, stock price trend prediction, and stock price rise or fall forecast
>
> - Login, registration, and logout functionalities
>
> - Testing and evaluation of the implemented components.

**Technical Stack**

> - Ecosystem: Python 3.8
>
> - Web Framework: Django 3
>
> - Data Storage Technology: Dataframe file storage
>
> - Deep Learning Framework: Keras
>
> - Frontend Technologies: bootstrap4 + jquery + ajax + echarts
>
> - Algorithms: LSTM, Normalization

**Structure introduction**

> - **stock**: The main app for the Django project.
>
> - **stockapp**: An app that implements stock display, forecasting, updating, and other operations.
>
> - **forecast**: An algorithm module that houses prediction algorithms, data processing functions, and various utility class functions.
>
> - **static**: Stores all kinds of static resources for the system (e.g., js, css).
>
> - **templates**: Holds frontend HTML pages.
> - **stockList.html**: The main page that displays a list of all stocks.
> - **stockDetail.html**: A stock detail page that showcases detailed information about a specific stock.
> - **stockSinglePredict.html**: A stock prediction page that displays predictions for a single stock, including stock trends, ranges, changes, and predictions for the next day's price.
> - **stockComparison.html**: A page for comparing information between two stocks.
> - **marketIndex.html**: Displays major market indices.
> - **data**: Contains `ts_code.csv` (a file that stores data for stocks with the code "ts_code") and `allStock.csv` (which stores brief information about all stocks).
>
> - **backup**: Stores temporary code files. It's redundant and can be deleted directly.


## To run the code

**preparation**
You need to get real-time data and API from a certain website:
tushare official site：[Tushare_data](https://www.tushare.pro/)
to get api token：[Tushare_data](https://www.tushare.pro/user/token)
AlphaVantage official site：[AlphaVantage_data](https://www.alphavantage.co/)
to get api token：[AlphaVantage_data](https://www.alphavantage.co/support/#api-key)

**Project Deployment**：
pip install --r requirements.txt

**Run**：
python manage.py runserver
