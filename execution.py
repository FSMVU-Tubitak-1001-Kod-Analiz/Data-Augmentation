import run
import pandas as pd
import numpy as np
import logic
import create_embedding as ce
import logic.augmentation as aug
from enum import Enum
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils.class_weight import compute_class_weight
import copy

data_file_name = "data/raw/unique_data_setV3.csv"
embedding_file_name ="data/extractions/embeddings/augmented_embeddings_graphcodebert_weight_new_loss_0_comof3.npy"
embedding_file_name_test ="data/extractions/embeddings/augmented_embeddings_graphcodebert_test.npy"
embedding_file_name_without_aug = "/home/user/PycharmProjects/Model_Scratch/data/graphcodebert_hidden_state.npy"

embedding_for_main_train = "data/extractions/embeddings/main_train_graphcodebert.npy"
embedding_for_main_test = "data/extractions/embeddings/main_test.npy"


parameters = {
    "optimizer": "Adam",
    "train_batch_size": 24,
    "lr": np.power(10., -4),
    "writer": SummaryWriter(),
    "seed": 42,
    "num_epochs": 300
}
device = "cuda:0"
class AugmentationType(Enum):
    CEILING = 1
    PERCENT = 2
    LCM = 3
class EmbeddingType(Enum):
    CODEBERT = 1
    GRAPHCODEBERT = 2
    NLIMEANS = 3
class EmbeddingSectionType(Enum):
    TRAIN = 1
    TEST=2


class Executor:
    def __init__(self,data_file_name,embedding_file_name,params,smells: int | tuple[int, int],embedding_file_name_test,aug_out_file_name ="new_data_set.csv",
                 only_refactored_file = "refactored.csv",weights = [],seperated_data_train_file_name ="train.csv",seperated_data_test_file_name ="test.csv",device = "cuda"):
        self.seperator = aug.CSVSeperator(data_file_name)
        self.seperated_data_train_file_name = seperated_data_train_file_name
        self.aug_util = aug.AugmentationUtils(self.seperated_data_train_file_name)
        self.seperated_data_test_file_name= seperated_data_test_file_name
        self.data_file_name = data_file_name
        self.embedding_file_name = embedding_file_name
        self.embedding_file_name_test = embedding_file_name_test
        self.aug_out_file_name= aug_out_file_name
        self.only_refactored_file =only_refactored_file
        self.weights= weights
        self.smells = smells
        self.device = device
        self.params = params
        #self.runner = run.Runner(data_file_name, embedding_file_name, device="cuda")


    def apply_ceiling_augmentation(self):
        self.augmentor.run_augmenting_as_ceiling()
    def apply_percent_augmentation(self):
        self.augmentor.run_augmenting_as_specified_weights()
    def apply_lcm_augmentation(self):
        self.augmentor.run_augmenting_as_lcm()

    def get_augmentation_coeffition(self,step,shift):
        selected_amount = self.smells if isinstance(self.smells,int) else max(self.smells)
        smells_count = self.aug_util.get_smells_count(selected_amount=selected_amount)
        smells_increment = self.aug_util.calculate_smells_increment(smells_count, step, shift=shift)
        coeffs= self.aug_util.get_coeff_list(smells_count, smells_increment, 2*step+1)
        return coeffs

    def embed_augmentation(self,embed_type : EmbeddingType,embed_section : EmbeddingSectionType):
        if embed_section == EmbeddingSectionType.TRAIN.value:
            if embed_type == EmbeddingType.CODEBERT.value:
                ce.create_embedding_codebert(self.aug_out_file_name, self.embedding_file_name)
            elif embed_type == EmbeddingType.GRAPHCODEBERT.value:
                ce.create_embedding_graphcodebert(self.aug_out_file_name, self.embedding_file_name)
            elif embed_type == EmbeddingType.NLIMEANS.value:
                ce.create_embedding_bert_nli_mean(self.aug_out_file_name, self.embedding_file_name)
        else:
            if embed_type == EmbeddingType.CODEBERT.value:
                ce.create_embedding_codebert(self.seperated_data_test_file_name,self.embedding_file_name_test )
            elif embed_type == EmbeddingType.GRAPHCODEBERT.value:
                ce.create_embedding_graphcodebert(self.seperated_data_test_file_name, self.embedding_file_name_test)
            elif embed_type == EmbeddingType.NLIMEANS.value:
                ce.create_embedding_bert_nli_mean(self.seperated_data_test_file_name, self.embedding_file_name_test)

    def seperate_csv(self,train,test):
        self.seperator.split_csv_as_percentage(0.2,train,test)

    def run_augmentation(self,aug_type:AugmentationType):
        self.seperator.split_csv_as_percentage(0.2,self.seperated_data_train_file_name,self.seperated_data_test_file_name)

        self.augmentor = aug.Augmentor(file_name=self.seperated_data_train_file_name, smells=self.smells, step=500,
                                       path_updated_smellKey=self.aug_out_file_name,only_refactored_file = self.only_refactored_file,
                                       weights=self.weights)
        if aug_type == AugmentationType.LCM.value:
            self.apply_lcm_augmentation()
        elif aug_type == AugmentationType.CEILING.value:
            self.apply_ceiling_augmentation()
        elif aug_type == AugmentationType.PERCENT.value:
            self.apply_percent_augmentation()


    def train_with_augmentation(self):

        self.runner = run.Runner(self.aug_out_file_name, self.embedding_file_name, 0.2, device=self.device,
                            params=self.params,
                            test_label_path=self.seperated_data_test_file_name,
                            test_file_path= (self.embedding_file_name_test))
        self.runner.run(self.smells)

    #loss penalty, loss = cross-entropy

    def train_without_augmentation(self):
        runner = run.Runner(self.data_file_name, self.embedding_file_name, 0.2, device=self.device,
                            params=self.params)
        runner.run(self.smells)

class ExecutionService:
    def __init__(self,executor : Executor):
        self.executor = executor
        self.path = 'data/'
        self.embedding_path = "extractions/embeddings/"

    def __base_func(self,func,cargs = []):
        self.executor.run_augmentation(AugmentationType.PERCENT.value)

        self.executor.embed_augmentation(embed_type=EmbeddingType.GRAPHCODEBERT.value,
                                     embed_section=EmbeddingSectionType.TRAIN.value)
        self.executor.embed_augmentation(embed_type=EmbeddingType.GRAPHCODEBERT.value,
                                     embed_section=EmbeddingSectionType.TEST.value)

        temp = self.executor.embedding_file_name
        temp2 = self.executor.embedding_file_name_test
        self.executor.embedding_file_name = self.path + self.executor.embedding_file_name
        self.executor.embedding_file_name_test = self.path + self.executor.embedding_file_name_test
        if len(cargs) != 0:
            func(cargs)
        else:
            func()
        self.executor.embedding_file_name = temp
        self.executor.embedding_file_name_test = temp2

    def base_model_execution(self):
        def best_model_exec_func():
            self.executor.train_with_augmentation()
        self.__base_func(best_model_exec_func)

    def base_model_execution_embedding_done(self):
        self.executor.train_with_augmentation()

    def base_model_with_penalty_execution(self, execution_time:int, select):

        def base_model_with_penalty_exec_func(cargs =[]):
            tempParam = dict(self.executor.params)
            select = cargs[1]
            for element in range(1, cargs[0]):
                df = pd.read_csv(self.executor.aug_out_file_name)
                class_weights = compute_class_weight('balanced', classes=select,
                                                     y=df.query("smellKey in @select")["smellKey"])
                self.executor.params["weights"] = class_weights * element
                print(class_weights, self.executor.params["weights"])
                self.executor.train_with_augmentation()
            self.executor.params = tempParam

        self.__base_func(base_model_with_penalty_exec_func,[execution_time,select])
    def base_model_with_penalty_execution_embedding_done(self, execution_time:int, select):
        tempParam = copy.deepcopy(self.executor.params)
        for element in range(1, execution_time):
            df = pd.read_csv(self.executor.aug_out_file_name)
            class_weights = compute_class_weight('balanced', classes=select,
                                                 y=df.query("smellKey in @select")["smellKey"])
            self.executor.params["weights"] = class_weights * element
            print(class_weights, self.executor.params["weights"])
            self.executor.train_with_augmentation()
        self.executor.params = tempParam


    def incremental_augmenting(self,step,shift,parameters):
        coeffs = self.executor.get_augmentation_coeffition(step, shift)
        self.executor.params = parameters

        print(coeffs)
        temp = self.executor.only_refactored_file
        temp2 = self.executor.aug_out_file_name
        temp3 = self.executor.embedding_file_name
        temp4 = self.executor.embedding_file_name_test

        for element in range(len(coeffs)):
            self.executor.only_refactored_file = str(element) +"_"+self.executor.only_refactored_file
            self.executor.aug_out_file_name = str(element) +"_"+ self.executor.aug_out_file_name
            self.executor.weights = coeffs[element]
            self.executor.run_augmentation(aug_type=AugmentationType.PERCENT.value)


            self.executor.embed_augmentation(embed_type=EmbeddingType.GRAPHCODEBERT.value,
                                        embed_section=EmbeddingSectionType.TRAIN.value)
            self.executor.embed_augmentation(embed_type=EmbeddingType.GRAPHCODEBERT.value,
                                        embed_section=EmbeddingSectionType.TEST.value)
            self.executor.embedding_file_name = self.path+ self.executor.embedding_file_name
            self.executor.embedding_file_name_test = self.path+ self.executor.embedding_file_name_test

            self.executor.train_with_augmentation()

            self.executor.only_refactored_file = temp
            self.executor.aug_out_file_name = temp2
            self.executor.embedding_file_name = temp3
            self.executor.embedding_file_name_test = temp4

# execution configuration trials
data_file_name = "data/raw/unique_data_setV3.csv"
embedding_file_name ="extractions/embeddings/augmented_embeddings_graphcodebert_weight_new_loss_0_comof3.npy"
embedding_file_name_test ="extractions/embeddings/augmented_embeddings_graphcodebert_test.npy"
parameters = {
    "optimizer": "Adam",
    "train_batch_size": 24,
    "lr": np.power(10., -4),
    "writer": SummaryWriter(),
    "seed": 42,
    "num_epochs": 300

}
device = "cuda:0"


deneme = ExecutionService(
Executor(data_file_name,embedding_file_name, parameters,6,embedding_file_name_test=embedding_file_name_test,device=device,
                    # only_refactored_file=only_refactored_file    #"weighted_refactored_new_loss_0_(comof3).csv"
                    # ,aug_out_file_name=aug_out_file_name         #"weighted_new_loss_0_new_data_set_(comof3).csv",
                    weights=[0,0,0,0,0,0])
)
# deneme.base_model_execution()


select = ["java:S100","java:S1161","java:S1172","java:S119","java:S1452","java:S3776"]
execution_time = 10
# deneme.base_model_with_penalty_execution(execution_time=execution_time,select= select)

step = 3
shift = 100
parameters = {
    "optimizer": "Adam",
    "train_batch_size": 48,
    "lr": np.power(10., -4),
    "writer": SummaryWriter(),
    "seed": 42,
    "num_epochs": 300

}
#deneme.incremental_augmenting(step,shift,parameters)

