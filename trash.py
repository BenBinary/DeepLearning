
    
array = pre_X[2]
max_X_2  = np.max(pre_X[2])
print("Predicition ", pre_X[2], "Test-Label ", y_test[2], "highest value: ", max_X_2, "Array Postiion", np.argmax(pre_X[2]))

print("Index: ", np.where(array == max_X_2));


# Ansatz mit index_min
index_max = np.argmax(array)
print("Index max ", index_max)


#print(numpy.amax())
print("Predicition ", pre_X[10], "Test-Label ", y_test[10], "highest value: ", max(pre_X[10]), "Array Postiion", np.argmax(pre_X[10]))    
print("Predicition ", pre_X[20], "Test-Label ", y_test[20], "highest value: ", max(pre_X[20]), "Array Postiion", np.argmax(pre_X[20]))    


# Test der Längen der Arrays
print("Länge Predicition ", len(pre_X))
print("Länge Labels ", len(y_test))
    # print(confusion_matrix(validation_generator.classes, y_pred))
