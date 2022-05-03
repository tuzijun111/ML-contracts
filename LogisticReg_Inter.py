from time import sleep
from web3 import Web3, HTTPProvider
import web3
import json
import datetime
import numpy as np
import copy
import pickle

class RopEth():
    def __init__(self):
        API_URL = "https://eth-ropsten.alchemyapi.io/v2/WIYEPYWikXeubjVSB1lnQB9RaZBa5oNP"
        PRIVATE_KEY = "7865c3bdd68113e704cea5a8fb002fda782005ab1a4680a7fbae31f01d320eb3"
        PUBLIC_KEY = "0x4f165218486CAE53022802701882b52d108076C3"
        self.contract_address = "0x80c6e9CdBA93e1f3Ca5c7eE32c9f1d17cCF89Be2"  #linear regression sc
        self.wallet_private_key = PRIVATE_KEY
        self.wallet_address = PUBLIC_KEY
        self.w3 = Web3(HTTPProvider(API_URL))
        with open('contracts/LogisticRegression.json') as f:
            self.data = json.load(f)
            # print("Contract ABI: ",self.data["abi"])
            self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.data["abi"])

    def model_gas_cost(self, train, test, l_rate, n_epoch):
        # Executing a transaction.
        nonce = self.w3.eth.get_transaction_count(self.wallet_address)
    
        estimatedGas1 = self.contract.functions.logistic_regression(train, test, l_rate, n_epoch).estimateGas()
        # print("Estimated gas to execute the transaction: ",estimatedGas)
        # print(dir(self.contract.functions.update(message)))
        txn_dict = self.contract.functions.logistic_regression(train, test, l_rate, n_epoch).buildTransaction({
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
    def incremental_model_train(self, train, test, l_rate, n_epoch, coef):
        # Executing a transaction.
        nonce = self.w3.eth.get_transaction_count(self.wallet_address)
    
        estimatedGas1 = self.contract.incremental_coefficients_sgd(train, test, l_rate, n_epoch, coef).estimateGas()
        # print("Estimated gas to execute the transaction: ",estimatedGas)
        # print(dir(self.contract.functions.update(message)))
        txn_dict = self.contract.incremental_coefficients_sgd(train, test, l_rate, n_epoch, coef).buildTransaction({
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


    def prediction_cost(self, item, coef):
        # Executing a transaction.
        nonce = self.w3.eth.get_transaction_count(self.wallet_address)
    
        estimatedGas1 = self.contract.functions.predict(item, coef).estimateGas()
        # print("Estimated gas to execute the transaction: ",estimatedGas)
        # print(dir(self.contract.functions.update(message)))
        txn_dict = self.contract.functions.predict(item, coef).buildTransaction({ # prediction for a single item
            'gas': estimatedGas1,
            'from': self.wallet_address,
            'nonce': nonce,
        })
        # print("gas for prediction:", estimatedGas1)
        
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

    def result_get(self):
        v = self.contract.functions.test().call()
        w = self.contract.functions.main().call()
        #print(dir(self.contract.functions()))
        print("The coefficients are: ", v)
        print("The predictions are: ", w)
    
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
        self.num_feature = 2
        self.num_input = 10 
        self.l_rate = 1*pow(10, 17)
        self.num_epoch = 2
        #self.model = "logistic regression"

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
        a_file = open("data/input_X.txt", "w")  
        np.savetxt(a_file, X)
        a_file.close()
        # store the Y values into a .txt file
        a_file = open("data/input_Y.txt", "w")
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
        a_file = open("data/feature_X.txt", "w")  
        np.savetxt(a_file, X)
        a_file.close()
        # store the Y values into a .txt file
        a_file = open("data/feature_Y.txt", "w")
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
        a_file = open("data/epoch_X.txt", "w")  
        np.savetxt(a_file, X)
        a_file.close()
        # store the Y values into a .txt file
        a_file = open("data/epoch_Y.txt", "w")
        np.savetxt(a_file, Y)
        a_file.close()
     
    # run the model without getting a subset of the dataset
    def model_run(self, dataset, num_epoch):
        r =RopEth()
        test = []
        train = dataset         
        for i in range(5):
            # test.append([])
            test.append(dataset[i])  
        r.model_gas_cost(train, test, self.l_rate, num_epoch)

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
        with open('data/dataset_input1000.txt', 'wb') as fp:
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
       

        

if __name__ == '__main__':
    m = MLmodel()
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
    dataset = [[2_7810836*pow(10, 11),  2_550537003*pow(10, 9), 0]]
    
    num_feature = 2
    num_input = 100
    num_epoch = 10
    # the initial coefficients for incremental training
    coef = [0] * (num_feature + 1)

    # create a big dataset
    # m.dataset_gen_txt(dataset, num_feature, num_input)
    
    with open ('dataset_input400.txt', 'rb') as fp:
        dataset = pickle.load(fp)

    print(len(dataset))
    print(len(dataset[0]))
    # print(np.loadtxt("input_X.txt", delimiter=","))
    
    m.prediction_test(dataset, coef)

    # m.model_run(dataset, num_epoch)

    # # m.num_of_feature_run(dataset, num_feature, num_epoch)

    # m.num_of_input_run(dataset, num_input, num_epoch)
    # m.num_of_epoch_run(dataset, num_epoch)



    # # call the incremental training method 
    # m.incremental_model_run(dataset, num_epoch, coef)


    

   


    
    # print("Transaction: ", t.hex())
    # r.getTransactionReciept(t)
    #print("Current data from contract: ", r.getLatestData())