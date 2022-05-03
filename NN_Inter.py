from time import sleep
from web3 import Web3, HTTPProvider
import web3
import json
import datetime
import numpy as np
import copy
import pickle
import random

class RopEth():
    def __init__(self):
        API_URL = "https://eth-ropsten.alchemyapi.io/v2/evXrvWnkYCOBdpZClctchytALhw50T_7"
        PRIVATE_KEY = "7865c3bdd68113e704cea5a8fb002fda782005ab1a4680a7fbae31f01d320eb3"
        PUBLIC_KEY = "0x4f165218486CAE53022802701882b52d108076C3"
        self.contract_address = "0x8ee6ae9E2B46CFAbfB818254a1bE7b979F28D772"  #linear regression sc
        self.wallet_private_key = PRIVATE_KEY
        self.wallet_address = PUBLIC_KEY
        self.w3 = Web3(HTTPProvider(API_URL))
        with open('contracts/NeuralNetwork.json') as f:
            self.data = json.load(f)
            # print("Contract ABI: ",self.data["abi"])
            self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.data["abi"])

    def model_gas_cost(self, network, train, label, l_rate, n_epoch, n_outputs):
        # Executing a transaction.
        nonce = self.w3.eth.get_transaction_count(self.wallet_address)
    
        estimatedGas1 = self.contract.functions.train_network(network, train, label, l_rate, n_epoch, n_outputs).estimateGas()
        
        # print("Estimated gas to execute the transaction: ",estimatedGas)
        # print(dir(self.contract.functions.update(message)))
        txn_dict = self.contract.functions.train_network(network, train, label, l_rate, n_epoch, n_outputs).buildTransaction({
            'gas': estimatedGas1,
            'from': self.wallet_address,
            'nonce': nonce,
        })
        print("gas for the whole model:", estimatedGas1)
        
        # # Need to sign to generate the transaction. Otherwise, we can not generate transactions on etherscan.
        # signPromise = self.w3.eth.account.signTransaction(txn_dict, self.wallet_private_key)
        # # Return the transaction hashvalue
        # result = self.w3.eth.send_raw_transaction(signPromise.rawTransaction)
        
        return estimatedGas1

    # since we can not train the model with large size dataset, we need to incrementally train the model
    # At each round, we feed the model with a part of the dataset

    def prediction_cost(self, network, row):
        # Executing a transaction.
        nonce = self.w3.eth.get_transaction_count(self.wallet_address)
    
        estimatedGas1 = self.contract.functions.forward_propagate(network, row).estimateGas()
        # print("Estimated gas to execute the transaction: ",estimatedGas)
        # print(dir(self.contract.functions.update(message)))
        txn_dict = self.contract.functions.forward_propagate(network, row).buildTransaction({ # prediction for a single item
            'gas': estimatedGas1,
            'from': self.wallet_address,
            'nonce': nonce,
        })
        print("gas for prediction:", estimatedGas1)
        
        # # Need to sign to generate the transaction. Otherwise, we can not generate transactions on etherscan.
        # signPromise = self.w3.eth.account.signTransaction(txn_dict, self.wallet_private_key)
        # # Return the transaction hashvalue
        # result = self.w3.eth.send_raw_transaction(signPromise.rawTransaction)
        
        return estimatedGas1
    

    def getTransactionReciept(self, txnHash):
        try:
            # sleep(400)
            response = self.w3.eth.wait_for_transaction_receipt( txnHash, timeout=120, poll_latency=0.1)
            print("True gas cost:", response)
            return response
        except web3.exceptions.TimeExhausted:
            print("what?")
            return None
        except web3.exceptions.TransactionNotFound:
            print("Something else")
        except Exception as ex:
            print("something other", ex)

    def result_get(self, network, train, label, l_rate, n_epoch, n_outputs):
        v = self.contract.functions.train_network(network, train, label, l_rate, n_epoch, n_outputs).call()
        # w = self.contract.functions.train_network( n_epoch, n_outputs).call()
        #print(dir(self.contract.functions()))
        print("The coefficients are: ", v)
        # print("The predictions are: ", w)
    
    def test_get(self, input):
    #def test_get(self, train, test, l_rate, n_epoch):
        v = self.contract.functions.test3(input).call()
        #v = self.contract.functions.logistic_regression(train, test, l_rate, n_epoch).call()
        print("result: ", v)

   
    def model_run(self, train, test, l_rate, n_epoch):
        v = self.contract.functions.logistic_regression(train, test, l_rate, n_epoch).call()
        print("result: ", v)
      
class MLmodel():
    def __init__(self):
        # the initial parameters of the dataset
        
        self.l_rate = 1*pow(10, 17)
        
        
        #self.model = "Neurual Network"

    def num_of_input(self, dataset, num_input, num_epoch):
        X = []
        Y = []
        r =RopEth()    
        # get the gas cost of the model with different (consecutive) parameters
        for k in range(self.num_input, num_input+1, 20):       
            # get the train dataset from the big dataset based on a specified number of features
            train = []
            test = []
            for i in range(k): 
                # train.append([])    
                train.append(dataset[i])   # since we need to add the label column to the dataset, so 
                # the totoal number of an item is k+1           
            for i in range(5):
                # test.append([])
                test.append(dataset[i])
            X.append(k)
            s = r.model_gas_cost(train, test, self.l_rate, num_epoch)
            Y.append(s)        

        return X, Y

    def num_of_input_run(self, dataset, num_input, num_epoch):
        X, Y = self.num_of_input(dataset, num_input, num_epoch)
        # store the X values into a .txt file
        a_file = open("input_X.txt", "w")  
        np.savetxt(a_file, X)
        a_file.close()
        # store the Y values into a .txt file
        a_file = open("input_Y.txt", "w")
        np.savetxt(a_file, Y)
        a_file.close()


    # return the data with different parameters. num_feature, num_input must not be less than them in the initial dataset
    # since we use the initial dataset as a basicline to augment the dataset
    def num_of_feature(self, dataset, num_feature, num_epoch):
        X = []
        Y = []
        r =RopEth()
        # get the gas cost of the model with different (consecutive) parameters
        for k in range(self.num_feature, num_feature+1, 5):       
            # get the train dataset from the big dataset based on a specified number of features
            train = []
            test = []
            for i in range(len(dataset)): 
                # train.append([])    
                train.append(dataset[i][0:(k+1)])   # since we need to add the label column to the dataset, so 
                # the totoal number of an item is k+1          
            for i in range(5):
                # test.append([])
                test.append(dataset[i][0:(k+1)]) 
            X.append(k)
            s = r.model_gas_cost(train, test, self.l_rate, num_epoch)
            Y.append(s)        
        return X, Y
    

    def num_of_feature_run(self, dataset, num_feature, num_epoch):
        X, Y = self.num_of_feature(dataset, num_feature, num_epoch)
        # store the X values into a .txt file
        a_file = open("feature_X.txt", "w")  
        np.savetxt(a_file, X)
        a_file.close()
        # store the Y values into a .txt file
        a_file = open("feature_Y.txt", "w")
        np.savetxt(a_file, Y)
        a_file.close()

    def num_of_epoch(self, dataset, num_epoch):
        X = []
        Y = [] 
        r =RopEth()  
        # get the gas cost of the model with different (consecutive) parameters
        for k in range(self.num_epoch, num_epoch+1):
            train = dataset
            test = []     
            for i in range(5):
                # test.append([])
                test.append(dataset[i])
            X.append(k)
            s = r.model_gas_cost(train, test, self.l_rate, k)
            Y.append(s)        
        return X, Y

    def num_of_epoch_run(self, dataset, num_epoch):
        X, Y = self.num_of_epoch(dataset, num_epoch)
        # store the X values into a .txt file
        a_file = open("epoch_X.txt", "w")  
        np.savetxt(a_file, X)
        a_file.close()
        # store the Y values into a .txt file
        a_file = open("epoch_Y.txt", "w")
        np.savetxt(a_file, Y)
        a_file.close()
     
    # run the model without getting a subset of the dataset
    def model_run(self, network, train, label, l_rate, n_epoch, n_outputs):
        r =RopEth()

        r.model_gas_cost(network, train, label, l_rate, n_epoch, n_outputs)

    def incremental_model_run(self, dataset, num_epoch, coef):
        r =RopEth()
        for k in range(0, len(dataset), 400):
            train = dataset[k:(k+400)]
            test = dataset[0:5]
            coef = r.incremental_model_train(train, test, self.l_rate, num_epoch, coef)
        return coef


    def dataset_gen_txt(self, dataset, num_feature, num_input):
        # increase the number of input data
        for i in range(num_input - 1): 
            dataset.append(copy.deepcopy(dataset[0]))
        # increase the number of features for each row
        for i in range(len(dataset)):
            for j in range(num_feature - self.num_feature):
                dataset[i].append(2_7810836*pow(10, 11))
        # return dataset    
        with open('dataset_input1000.txt', 'wb') as fp:
            pickle.dump(dataset, fp)
        # np.savetxt(a_file, dataset, fmt='%s')
        # a_file.close()

    def prediction_test(self, dataset, coef):
        r = RopEth()
        
        
        item = []
        coef = [1*pow(10, 11)] * (3)
        Y = []
        for i in range(10, len(dataset), 20):
            item = dataset[i]
            res = r.prediction_cost(item, coef)
            Y.append(res)

        # store the Y values into a .txt file
        a_file = open("prediction_input_Y.txt", "w")
        np.savetxt(a_file, Y)
        a_file.close()


    # Initialize a network
    def initialize_network(self, n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [[random.randint(10, 99)*pow(10, 16) for i in range(n_inputs )] for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [[random.randint(10, 99)*pow(10, 16) for i in range(n_hidden )] for i in range(n_outputs)]
        network.append(output_layer)
        return network
       

        

if __name__ == '__main__':
    import numpy as np
    
    # initial dataset
    # dataset = [[2_7810836e+11,  2_550537003e+9, 0],
    #            [1_465489372e+9, 2_362125076e+9, 0],
    #            [3_396561688e+9, 4_400293529e+9, 0],
    #            [1_38807019e+10, 1_850220317e+9, 0],
    #            [3_06407232e+10, 3_005305973e+9, 0],
    #            [7_627531214e+9, 2_759262235e+9, 1e+18],
    #            [5_332441248e+9, 2_088626775e+9, 1e+18],
    #            [6_922596716e+9, 1_77106367e+10, 1e+18],
    #            [8_675418651e+9, -242068655e+8,  1e+18],
    #            [7_673756466e+9, 3_508563011e+9, 1e+18]]
    # dataset = [[2_7810836*pow(10, 11),  2_550537003*pow(10, 9), 0]]
    
    
    l_rate = 1*pow(10, 17)
    n_epoch = 2
    
    n_inputs = 3   # depends on the number of feature of input dataset
    n_hidden = 2
    n_outputs = 2

    
    # create a big dataset
    # m.dataset_gen_txt(dataset, num_feature, num_input)
    m = MLmodel()
    r =RopEth()
    
    with open ('data/dataset_feature.txt', 'rb') as fp:
        dataset = pickle.load(fp)
    
    # print(len(dataset))
    # print(dataset[0])
    label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    train = dataset[:4]

    X = []
    Y = []
    # row = dataset[0][:10]
    # for i in range(1, 10):
    #     m = MLmodel()
    #     network = m.initialize_network(n_inputs, n_hidden, i)
    #     x = i
    #     y = r.prediction_cost(network, row)
    #     X.append(x)
    #     Y.append(y)

    # a_file = open("NN/prediction_hidden_X.txt", "w")
    # np.savetxt(a_file, X)
    # a_file.close()
    # b_file = open("NN/prediction_hidden_Y.txt", "w")
    # np.savetxt(b_file, Y)
    # b_file.close()
    
    
    for i in range(2, 10):
        m = MLmodel()
        network = m.initialize_network(i, n_hidden, n_outputs)
        y = m.model_run(network, train, label, l_rate, n_epoch, n_outputs)
        
        Y.append(y)

    
    b_file = open("NN/infer_hidden_Y.txt", "w")
    np.savetxt(b_file, Y)
    b_file.close()
    
    # network = m.initialize_network(n_inputs, n_hidden, n_outputs)
    # m.model_run(network, train, label, l_rate, n_epoch, n_outputs)

    
   
    
    # for j in range(len(1, n_hidden)):
    #     for m in range(len(1, n_outputs)):
            
            
    #             m = MLmodel()
    #             network = m.initialize_network(n_inputs,j,m)
    #             res = r.prediction_cost(network, row)
    #             Y.append(res)
    # a_file = open("NN/prediction_weight_Y.txt", "w")
    # np.savetxt(a_file, Y)
    # a_file.close()
    

    
   
    # m.model_run(network, train, label, l_rate, n_epoch, n_outputs)
    

    
    # r.model_gas_cost(network, train, label, l_rate, n_epoch, n_outputs)
    # r.prediction_cost(network, row)
    # r.result_get(network, train, label, l_rate, n_epoch, n_outputs)
    

    # model_gas_cost(self, network, train, label, l_rate, n_epoch, n_outputs):
    # model_run(self, network, train, label, l_rate, n_epoch, n_outputs):


    

   


    
