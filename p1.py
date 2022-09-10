import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import metrics

def predict(ax, data, sales_data, predictor_column):

    predictor = pd.DataFrame(data, columns=[predictor_column])[:20]
    test_data = pd.DataFrame(data, columns=[predictor_column])[20:]
    
    # Select a linear model
    model = sklearn.linear_model.LinearRegression()

    # Train the model
    model.fit(predictor.values, sales_data.values[:20])

    # Make a prediction
    predict_data = model.predict(test_data.values)

    plt.plot(predictor.values, sales_data.values[:20], 'o', label='Training Data')
    plt.plot(test_data.values, predict_data, 'o', label='Predicted Data')
    plt.legend(loc='best')

    RMSE = np.sqrt(metrics.mean_squared_error(test_data.values, predict_data))
    # print(f'RMSE: {RMSE}')
    ax.set_title(predictor_column + " => RMSE: " + str(RMSE))
    
def main():
    # Load and prepare the data
    data = pd.read_excel('mlr05.xls', 'Mlr05')
    sales_data = pd.DataFrame(data, columns=['X1'])
    predictor_col = ["X2", "X3", "X4", "X5", "X6"]
    
    ncols = 2
    nrows = len(predictor_col) // ncols + (len(predictor_col) % ncols > 0)
    
    fig = plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    for col in predictor_col:
        ax = fig.add_subplot(nrows, ncols, predictor_col.index(col) + 1)
        predict(ax, data, sales_data, col)
    
    plt.savefig('p1.png')
    plt.show()

if __name__ == '__main__':
   main()