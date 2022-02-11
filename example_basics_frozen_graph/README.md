# Basic example
This is a very basic example wich demonstrates how to load and infer a _Frozen Graph_ created using TensorFlow.
_Frozen Graph_ is the standard of storing a graph for versions of TensorFlow < 1.15.
We wanted to support this feature for legacy reasons.

### TensorFlow2
We will create a dummy graph for demonstration purposes.
To create the model simply run the following command:
```
python main.py
```

### openFrameworks
By default `ofxTF2::Model` will use the `SavedModel` format. To use the `FrozenGraph` format you may either add the type to the constructor or call `setModelType()` afterwards.
As the default names differ from the names in the `SavedModel` format make sure to overwrite names of the ins and outs by calling `setup()`.
```c++
// set model type and i/o names
model.setModelType(cppflow::model::TYPE::FROZEN_GRAPH);
model.setup({{"x:0"}}, {{"Identity:0"}});
```
Afterwards you can load the _pb file_ using the `load()` function.
```c++
// load the model, bail out on error
if(!model.load("model.pb")) {
	std::exit(EXIT_FAILURE);
}
```

Everything else should work the same.

Please understand that we wont be able to invest a lot of time in supporting this feature in the future.