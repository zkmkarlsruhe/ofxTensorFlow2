# Basic example
This example is a proof of concept and demonstrates how to evaluate a 
_SavedModel_ with multiple in and outputs.

### TensorFlow2
We dont want to go into details about the training process of such a model. For 
anyone interested in the the topic: the code has been taken from 
[this example](https://www.tensorflow.org/guide/keras/functional).
For this reason we will only create a computational graph. Keras' functional 
model API is best suited for this type of architecture.
Start by defining the input tensors. They may have different shapes and/or dtype.
```python
title_input = keras.Input(shape=(None,), name="title")
body_input = keras.Input(shape=(None,), name="body")
tags_input = keras.Input(shape=(12,), name="tags")
```
Each input may then be processed separately, but will eventually be concatenated
with other parts of the model to form a single input to the next layer(s).
```python
x = layers.concatenate([title_features, body_features, tags_input])
```
The concatenated inputs are here processed by two different fully-connected layers. Those will represent our output tensors.
```python
priority_pred = layers.Dense(1, name="priority")(x)
department_pred = layers.Dense(num_departments, name="department")(x)
```
Finally, we define the model.
```python
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred])
```
As we are omiting the training process we export the model already at this stage.
```python
model.save('../bin/data/model')
```

### openFrameworks
In openFrameworks we start by instanciating an `ofxTF2::Model`.
```C++
ofxTF2::Model model;
model.load("model");
```
Afterwars we define the in and output names of our model as a vector of strings.
These are then passed to the `setup()` function.
***Note***: you can always check the signature using the CLI tool 
`saved_model_cli`, e.g. `saved_model_cli show --dir path/to/model/ --tag_set serve --signature_def serving_default`.
```C++
std::vector<std::string> inputNames = {
	"serving_default_body",
	"serving_default_tags",
	"serving_default_title"
};
std::vector<std::string> outputNames = {
	"StatefulPartitionedCall:0",
	"StatefulPartitionedCall:1"
};
model.setup(inputNames, outputNames);
```
Next we define our inputs and wrap them __in the same order__ into a vector of 
tensors.
```C++
cppflow::tensor inputBody = cppflow::fill({1, 2}, 2.0f);
cppflow::tensor inputTags = cppflow::fill({1, 12}, 1.0f);
cppflow::tensor inputTitle = cppflow::fill({1, 3}, 4.0f);

std::vector<cppflow::tensor> vectorOfInputTensors = {
		inputBody, inputTags, inputTitle
};	
```
Now you can infer the model using `runMultiModel()` and retrieve the individual 
output tensors using the [ ] operator.
```C++
std::vector<cppflow::tensor> vectorOfOutputTensors = 
	model.runMultiModel(vectorOfInputTensors);

cppflow::tensor outputPrio = vectorOfOutputTensors[0];
cppflow::tensor outputDept = vectorOfOutputTensors[1];
```

***Note***: the same pattern applies to classes that are derived from `Model` 
such as `ThreadedModel`. 

***Note***: executing `runModel` will call `runMultiModel` with single in and 
output tensors. By default, the model uses the input name "serving_default_1" 
and the output name "StatefulPartitionedCall".