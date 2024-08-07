# Data-Augmentation

This repository has been constructed with the specific purpose of facilitating code data augmentation and this repository used in article named [article_name]. It serves as a centralised repository for a variety of refactoring techniques for training process, which are employed with the objective of enhancing the diversity and volume of code datasets. The method employed a weight factor to augment the data volume. In the learning process, the options provided were limited to the base model, the base model with loss penalty, and incremental augmentation, as detailed in the article.

# Initialization

All logic can be executed from execution.py. In order to execute the code, it is necessary to initialise and run the ExecutionService class. The following section illustrates how this can be done in a specific case:

Firstly, suitable parameters must be set before execution

```python
data_file_name = "data/raw/dataset.csv"
embedding_file_name ="extractions/embeddings/embedding.npy"
embedding_file_name_test ="extractions/embeddings/embedding_test.npy"
parameters = {
    "optimizer": "Adam",
    "train_batch_size": 24,
    "lr": np.power(10., -4),
    "writer": SummaryWriter(),
    "seed": 42,
    "num_epochs": 300
}
device = "cuda:0"


trial = ExecutionService(
Executor(data_file_name,embedding_file_name, parameters,6,embedding_file_name_test=embedding_file_name_test,device=device,
                    # only_refactored_file=    #"refactored_samples.csv",
                    # aug_out_file_name=      #"original_samples_and_refactored_samples.csv",
                    weights=[0,0,0,0,0,0])
)

```

to execute only base model without utilizing augmentation:

```python
trial.base_model_execution()

```

to execute model with loss penalty:

```python
select = ["java:S100","java:S1161","java:S1172","java:S119","java:S1452","java:S3776"] #This names are label classes.
execution_time = 10 #loop amount of training from the beginning
trial.base_model_with_penalty_execution(execution_time=execution_time,select= select)

```

to execute model with incremental augmentation:

```python
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
trial.incremental_augmenting(step,shift,parameters)

```

All of the aforementioned details can be located within the execution.py file. Those seeking to create a custom initialization process are advised to utilise the Executor class within execution.py. The following section will present a comprehensive overview of all the requisite information pertaining to the Executor class modules.

# Executor

This class contains initialization logic

```python
executor =Executor(data_file_name,
                embedding_file_name,
                params,
                smells: int | tuple[int, int],
                embedding_file_name_test,
                aug_out_file_name ="new_data_set.csv",
                only_refactored_file = "refactored.csv",
                weights = [],
                seperated_data_train_file_name ="train.csv",
                seperated_data_test_file_name ="test.csv",
                device = "cuda"):
```

-   **data_file_name:** Parameter is the designation of the dataset in the .csv format.
-   **embedding_file_name:** Parameter is the designation of the embedding of train dataset in the .npy format.
-   **params:** Parameter stores learning related parameters such as optimizer, learning rate, epoch, batch size... All necessary parameter is displayed above examples.
-   **smells:** Parameter accepts an integer or a tuple of integers as input and specifies the maximum amount of class labels in descending order that are processed by the logic.
-   **embedding_file_name_test:** Parameter is the designation of the embedding of test dataset in the .npy format..
-   **aug_out_file_name:** This parameter created a .csv file with specified name. This file contains both the original data and the augmented data.
-   **only_refactored_file** This paraemeter created a .csv file with speficied name.This file contains only the augmented data.
-   **weights:** Takes a list that specify percentage of one class label incrementation amount. For instance, when class labels with [100,200,300] samples is inputted [1.0,0.5,0.5], it will result class labels with [200,300,450] samples.
-   **seperated_data_train_file_name:** This paraemeter created a .csv file with speficied name.This file contains only training data set for learning process
-   **seperated_data_test_file_name:** This paraemeter created a .csv file with speficied name.This file contains only test data set for learning process
-   **device:** Specify whether use cpu or cuda.

**Necessity for Proper Initialization**

1. Run augmentation process. If you dont need any augmentation, seperate your dataset as train set and test set
2. Create embeddings for your train set and test set.
3. Set your embedings path. Embeddings is added into "data/...(your directory or file)", so you should set the path completely.
4. Do your personal configuration and start training.
5. Create a Runner and run.

An example:

```python
    import run
    path_start = "data/"
    def example_initialization(self,callback):
        executorInstance = Executor(...)
        executorInstance.run_augmentation(AugmentationType.PERCENT.value)

        executorInstance.embed_augmentation(embed_type=EmbeddingType.GRAPHCODEBERT.value,
                                     embed_section=EmbeddingSectionType.TRAIN.value)
        executorInstance.embed_augmentation(embed_type=EmbeddingType.GRAPHCODEBERT.value,
                                     embed_section=EmbeddingSectionType.TEST.value)

        executorInstance.embedding_file_name_path = path_start + executorInstance.embedding_file_name
        executorInstance.embedding_file_name_test_path = path_start + executorInstance.embedding_file_name_test

        executorInstance.runner = run.Runner(executorInstance.aug_out_file_name,
                            executorInstance.embedding_file_name_path,
                            0.2,
                            device=executorInstance.device,
                            params=executorInstance.params,
                            test_label_path=executorInstance.seperated_data_test_file_name,
                             test_file_path= (executorInstance.embedding_file_name_test_path))
        executorInstance.runner.run(executorInstance.smells)



    example_initialization()
```

-   WARNING: Executor class methods uses embedding_file_name and embedding_file_name_test variable as Runner class parameter. So you must speficy it value not only file name but adding its path. For instance, instead of "a.npy", it must be "data/a.npy"

# Modules for Customization

**Augmentor**

<hr/>
This class contains all business logic of augmentation. All explanation of parameters is below

-   **file_name:** Parameter is the designation of the dataset in the .csv format.
-   **smells:** Parameter accepts an integer or a tuple of integers as input and specifies the maximum amount of class labels in descending order that are processed by the logic.
-   **step:** Specify amount of data is loaded in single loop
-   **weights:** Takes a list that specify percentage of one class label incrementation amount. For instance, when class labels with [100,200,300] samples is inputted [1.0,0.5,0.5], it will result class labels with [200,300,450] samples.
-   **path_updated_smellKey:** This parameter created a .csv file with specified name. This file contains both the original data and the augmented data.
-   **only_refactored_file** This paraemeter created a .csv file with speficied name.This file contains only the augmented data.

The class offers three distinct options for utilisation. These are the LCM (least common factor), ceiling, and class weight options.

```python
augmentor = aug.Augmentor(file_name, smells, step=500, path_updated_smellKey,only_refactored_file,weights)

augmentor.run_augmenting_as_ceiling()

augmentor.run_augmenting_as_specified_weights()

augmentor.run_augmenting_as_lcm()

```

**CSVSeperator**

<hr/>

The class contains the separation of the data set into a training set and a test set. All explanations of the parameters are provided below:

-   **file_name:** name of the dataset as .csv format

This class has one methods:

```python
seperator = aug.CSVSeperator(data_file_name)

seperator.split_csv_as_percentage(percentage,
                                train_file_name="train.csv",
                                test_file_name="test.csv",
                                shuffle = False)

```

-   percentage: Parameter accepts values between 0.0 and 1.0, indicating the proportion of the test sample.
-   train_file_name: Paremeter designate the names of the new train samples
-   test_file_name: Paremeter designate the names of the new test samples
-   shuffle: Parameter specifies whether the original data should be shuffled.

**AugmentationUtils**

<hr/>

The class contains all class weight coefficient calculation logic for incremental augmentation process.All explanations of the parameters are provided below:

-   **file_name:** name of the dataset as .csv format

To utilize this class functions:

```python

class_label_amount = 6
step = 3 # specify iteration amount when all each class samples is equal.
shift =100 # specify sample increment amount of class with maximum sample

aug_util = AugmentationUtils("training_file_name.csv")

smells_count = aug_util.get_smells_count(selected_amount=class_label_amount)
smells_increment = aug_util.calculate_smells_increment(smells_count, step, shift=shift)
coeffs= aug_util.get_coeff_list(smells_count, smells_increment, 2*step+1)

print(coeffs)

```

**Runner**

<hr/>

The class contains learning process.All explanations of the parameters are provided below:

-   **label_path:** Parameter is the designation of the training dataset in the .csv format.
-   **file_path:** Parameter is the designation of the embedding of train dataset in the .npy format.
-   **val_ratio:** Validation ratio.
-   **device:** Specify whether use cpu or cuda.
-   **params:** Parameter stores learning related parameters such as optimizer, learning rate, epoch, batch size... All necessary parameter is displayed above examples.
-   **test_label_path:** This paraemeter created a .csv file with speficied name.This file contains only test data set for learning process
-   **test_file_path:** Parameter is the designation of the embedding of test dataset in the .npy format..

to execute Runner class instance:

```python
import run

runner = run.Runner(...)

smells = 5
runner.run(smells, shuffle= False)
```
