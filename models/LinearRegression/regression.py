from sklearn.metrics import mean_squared_error, r2_score
import dataset.dataset as d_io
import dataset.utility as util
from sklearn import linear_model


frame = d_io.read_imputed_onehot_dataset()

print("Handoff started")

x_training = util.get_frame_in_range(frame, 3, 2016, 12, 2017)
x_testing = util.get_frame_in_range(frame, 1, 2018, 2, 2018)
x_training = d_io.to_numpy(x_training.drop(columns=['NumberOfCustomers', 'Date']))
x_testing = d_io.to_numpy(x_testing.drop(columns=['NumberOfCustomers', 'Date']))

y_training = d_io.to_numpy(util.get_frame_in_range(frame, 3, 2016, 12, 2017).NumberOfSales)
y_testing = d_io.to_numpy(util.get_frame_in_range(frame, 1, 2018, 2, 2018).NumberOfSales)

print("Training has %s samples, testing has %s samples" % (len(y_training), len(y_testing)))

print("Starting regression")

regression = linear_model.LinearRegression()
regression.fit(x_training, y_training)

print("Done fitting")

# Make predictions using the testing set
prediction = regression.predict(x_testing)

print("REAL %s\n PRED = %s" % (y_testing, prediction))

# The coefficients
print('Coefficients: \n', regression.coef_)
# The mean squared error
print("Mean squared error: %s"
      % mean_squared_error(y_testing, prediction))
# Explained variance score: 1 is perfect prediction
print('Variance score: %s' % r2_score(y_testing, prediction))

