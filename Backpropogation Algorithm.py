#impotring packages
import random
from math import exp
from random import seed

#initializing the network
def initialize_network(n_inputs,n_hidden,n_outputs):
    network=list()
    hidden_layer=[{'weights':[random.uniform(-0.5,0.5) for i in range(n_inputs+1)]}
                  for i in range (n_hidden)] 
    network.append(hidden_layer)
    output_layer=[{'weights':[random.uniform(-0.5,0.5) for i in range (n_hidden+1)]} 
                  for i in range (n_outputs)] 
    network.append(output_layer)
    return network

  #Calculate neuron activation for a input
def activate(weights,inputs):
    activation=weights[-1]
    for i in range (len(weights)-1):
        activation+=weights[i]*inputs[i]
    return activation

#Transfer neural activation
def transfer(activation):
    return 1.0/(1.0+exp(-activation))

#Forward propogate inout to a network output
def forward_propogate(network,row):
    inputs=row
    for layer in network :
        new_inputs=[]
        for neuron in layer:
            activation=activate(neuron['weights'],inputs)
            neuron['output']=transfer(activation)
            new_inputs.append(neuron['output'])
        inputs=new_inputs
    return inputs

#Calculate derivative of an neuron output
def transfer_derivative(output):
    return output*(1.0-output)

#Backpropogate error and store in neuron
def backward_propogate_error(network,expected):
    for i in reversed(range(len(network))):
        layer=network[i]
        errors=list()
        if i!=len(network)-1:
            for j in range(len(layer)):
                error=0.0
                for neuron in network[i+1]:
                    error+=(neuron['weights'][j]*neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron=layer[j]
                errors.append(expected[j]-neuron['output'])
        for j in range (len(layer)):
            neuron=layer[j]
            neuron['delta']=errors[j]*transfer_derivative(neuron['output'])

#Update network weights with error
def update_weights(network,row,l_rate):
    for i in range(len(network)):
        inputs=row[:-1]
        if i!=0:
            inputs=[neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=l_rate*neuron['delta']*inputs[j]
                neuron['weights'][-1]+=l_rate*neuron['delta']
#Train network for a fixed number of epochs
def train_network(network, train,l_rate, n_epoch,n_outputs):
    for epoch in range(n_epoch):
        sum_error=0
        for row in train:
            outputs = forward_propogate(network,row)
            expected=[0 for i in range(n_outputs)]
            expected[row[-1]]=1
            sum_error+=sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propogate_error(network,expected)
            update_weights(network,row,l_rate)
        print('>=epoch=%d ,lrate=%3f,error=%3f' %(epoch,l_rate,sum_error))

#Test training back propogate algorithm
seed(1)
dataset = [[2.7810836,2.550537003,0],
          [1.465489372,2.362125076,0],
          [3.396561688,4.400293529,0],
          [1.38807019,1.850220317,0],
          [3.06407232,3.005305973,0],
          [7.627531214,2.759262235,1],
          [5.332441248,2.0883626775,1],
          [6.922596716,1.77105362,1]]
n_inputs=len(dataset[0])
n_outputs=len(set([row[-1] for row in dataset]))
network=initialize_network(n_inputs,2,n_outputs)
train_network(network,dataset,0.5,20,n_outputs)

#Output
>=epoch=0 ,lrate=0.500000,error=4.125020
>=epoch=1 ,lrate=0.500000,error=3.968198
>=epoch=2 ,lrate=0.500000,error=3.866199
>=epoch=3 ,lrate=0.500000,error=3.780908
>=epoch=4 ,lrate=0.500000,error=3.688478
>=epoch=5 ,lrate=0.500000,error=3.575438
>=epoch=6 ,lrate=0.500000,error=3.434408
>=epoch=7 ,lrate=0.500000,error=3.262814
>=epoch=8 ,lrate=0.500000,error=3.062552
>=epoch=9 ,lrate=0.500000,error=2.839223
>=epoch=10 ,lrate=0.500000,error=2.601103
>=epoch=11 ,lrate=0.500000,error=2.357944
>=epoch=12 ,lrate=0.500000,error=2.119421
>=epoch=13 ,lrate=0.500000,error=1.893597
>=epoch=14 ,lrate=0.500000,error=1.685996
>=epoch=15 ,lrate=0.500000,error=1.499473
>=epoch=16 ,lrate=0.500000,error=1.334664
>=epoch=17 ,lrate=0.500000,error=1.190671
>=epoch=18 ,lrate=0.500000,error=1.065726
>=epoch=19 ,lrate=0.500000,error=0.957693



#For layer in network : Print (layer)
i=1
for layer in network:
    j=1
    for sub in layer:
        print("\nLayer[%d] Node[%d]:\n" %(i,j),sub)
        j=j+1
    i=i+1
#For layer in network : Print (layer)


#Final output
Layer[1] Node[1]:
 {'weights': [-0.9500421875225666, 1.175711645054104, 0.26377461897661403, 0.2878952930454144], 'output': 0.019981479193988023, 'delta': -0.0018875216918754884}

Layer[1] Node[2]:
 {'weights': [1.13909780838039, -1.4889498808090191, 0.15159297272276295, -0.6416275012460004], 'output': 0.9911957501702244, 'delta': 0.0013114704584803246}

Layer[2] Node[1]:
 {'weights': [1.0129238516656542, -1.9396170061806988, 0.6647705887005229], 'output': 0.23613148120509259, 'delta': -0.04259183924282797}

Layer[2] Node[2]:
 {'weights': [-1.3565766903832266, 1.7674699457216334, -0.4293847613832582], 'output': 0.775026352541806, 'delta': 0.03922651887419075}
