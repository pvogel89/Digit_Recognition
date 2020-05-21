

def prepare_data(dataset):
    print('Start to Prepare the Training/Validation Data')
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return [x_train, y_train, x_test, y_test]
