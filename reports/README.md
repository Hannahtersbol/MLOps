# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [X] Setup version control for your data or part of your data (M8)
* [X] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [X] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [X] Use profiling to optimize your code (M12)
* [X] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X] Write unit tests related to the data part of your code (M16)
* [X] Write unit tests related to model construction and or model training (M16)
* [X] Calculate the code coverage (M16)
* [X] Get some continuous integration running on the GitHub repository (M17)
* [X] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [X] Add a linting step to your continuous integration (M17)
* [X] Add pre-commit hooks to your version control setup (M18)
* [X] Add a continues workflow that triggers when data changes (M19)
* [X] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [X] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [X] Create a trigger workflow for automatically building your docker images (M21)
* [X] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 9

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s224758, s224775, s224762

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the framework called 'Timm: Pytorch Image Models'. Timm is a collection of models, both pretrained and not, that are suited for image classification.
Therefore it fit well with our project about cat/dog classification. We used their "resnet-18" model that was recommended for this type of task.
We just had to add an extra layer at the of the model as a new output layer, as we only wanted 2 output nodes: Cat/Dog. This model also enabled you to use
a variable amount of input channels. As we had decided to use a greyscale 150x150 image-dataset, it was very handy for us that we could reduce the models
expected input channels from 3 to 1.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We have used pip to manage dependencies in our project. The dependencies have been listed in the requirements.txt file, while development-specific dependencies were set into requirements_dev.txt. Both files contain the versions or reference, preventing possibe issues tied to different versions of the extentions. The production file includes essential libraries like torch, fastapi, and google-cloud-storage, while the development file adds tools for testing, linting, and documentation, such as pytest and mkdocs.

To replicate the environment, a new team member would:

Clone the project repository.
Activate an environment.
Install production dependencies with pip install -r requirements.txt.
If working on development, install additional packages from requirements_dev.txt with pip install -r requirements_dev.txt.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We have used the cookiecutter template but we removed the docs and notebook folder as we didn't use them.
In our data folder we split it up in a cat and dog folder as our dataset didn't come with explicit labels.
So instead we used the path of the folder, from which the picture came, to create the labels for them.
We added an outputs folder that contains the logs from hydra, so we could log each experiment we did,
so even if we changed an experiment-config-file, we would still have the old experiments stored.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

To keep our code clean and readable, we used Ruff for linting and formatting. This tool is great for teamwork because it makes sure all our code looks the same, which helps us understand each other's work more easily.

We also added type annotations to some of our Python methods. This helps us catch mistakes early,
especially when using multiple frameworks. By keeping our code consistent,
we make it more reliable and easier to maintain. This approach also makes debugging simpler,
as it’s easier to spot and fix issues. Overall, these practices improve our development process and enhance code quality.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented 9 tests across three areas.

- API Tests (2 tests): These validate the /preprocess endpoint to ensure proper handling of query parameters and the /evaluate-image    endpoint for correct classification and integration with the model using mocked functions.
- Data Tests (5 tests): These confirm dataset integrity, including verifying the dataset length matches expectations, transformations are defined, and the shapes of transformed cat and dog images are consistent. Additionally, we ensure the dataset reads from the correct directory.
- Model Tests (3 tests): These include checking the forward pass for correct output shape, validating the presence of trainable parameters, and ensuring the model’s structure matches the expected configuration.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

In the beginning we didn't use a lot of branches. This was deliberate as all of us were unsure of the project setup.
Instead we started with mob-programming, where 1 person is coding while the rest are directing them on what to write.
When we had the main functionality of the framework, models and data sorted, we up our work.
We used seperate branches when we were working on features that affected already established functionality.
This relates mainly to our python code, as many members could be working on that simultainiously.
Features like github actions could still be worked on the main branch as only 1 person was assigned to those features.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

In the beginning we didn't use a lot of branches. This was deliberate as all of us were unsure of the project setup.
Instead we started with mob-programming, where 1 person is coding while the rest are directing them on what to write.
When we had the main functionality of the framework, models and data sorted, we up our work.
We used seperate branches when we were working on features that affected already established functionality.
This relates mainly to our python code, as many members could be working on that simultainiously.
Features like github actions could still be worked on the main branch as only 1 person was assigned to those features.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:
We did set up dvc, but did not end up using it too much. It wasn't feasible to have all the data in our git repository
and push/pull it every time, so we set up dvc to push to a remote branch, which is a public storage bucket hosted on gcloud storage.
In this way it helped us to move vast amounts of data around, but we did not use the version control aspects of it,
because we did not change the data or do any cleanup. Data version control becomes very important when you change data,
like removing wrong training data or in other ways manipulating it.
Without version control it becomes impossible to reproduce the models that were based on previous data.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have made pytests in order to ensure that if we were to change anything the test would catch any error during the implementation.
We have made tests for the data: testing the length of the datasets, the format and shape of the data,
and the path of datasets all to ensure that we are readion the correct data and it is implemented properly.
We have also made tests for the model. Testing the output shape, number of parameters and the generel structure of the model to ensure that it also works as intended.
We have also made tests for the api to ensure that it evaluates preprocess properly and evaluates an image properly.
This is to ensure that there is no mistake before using these api function now that there can be a lot of different mistakes
when it comes to passing object or information though the paths and whether it is a post or get api function.
We have also done some pre commits that checks syntax and formatting to avoid pushing faulty code to the git

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used hydra combined with config.yaml files. To run an experiment we would write:
invoke train -x Exp1
invoke evaluate -m M_Exp1
This would train a model based on the hyperparameters located in configs/Exp1.yaml
and then evaluate the outputted model found at models/M_Exp1.pth
Example of config file:
#config.yaml
info:
  name: Exp1

hyperparameters:
  batch_size: 64
  learning_rate: 1e-4
  epochs: 10
  seed: 42

After the experiment, hydra would then log the experiment, including the hyperparameters, in log/
So even if we changed a config file, we would still be able to look at old configurations.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

So as stated previously, hydra was used to keep track of every experiment done. We made sure that our randomization seed was also
a hyperparameter, so that generated random numbers would be the same if you tried to reproduce an experiment. The next problem is that
two different machines can get two different results. To remedy this we use docker to isolate the dependencies and containerize them.
With docker we ensure that everything is identical when our experiments if we use the the same docker images. This is crucial to be
able to analyze real world models and detect their weaknesses, which needs to happen before you can fix and improve them.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

Experiment 1: [this figure](figures/training_statistics_Exp1.png)
Experiment 2: [this figure](figures/training_statistics_Exp2.png)

Evaluations: [this figure](figures/evaluations.png)

We tracked the same statistics on two experiments. In experiment 2 we doubled the learning rate. This increased

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging: While debuuging have varied from person to person, some repeated practises have been to use the error messages from the Terminal when running the code. As well as using ChatGPT and GitHub's copilot to help solve the issues which arose. Furthermore we tried using the python debugging tool showed in the curse. That being said we also occasionally relied on print statements—an old habit which, while not always ideal, still provided some good insights.

Profiling: We performed profiling on our code, which initially revealed that the training phase spent most of its time moving data rather than executing the training functions. Based on this insight, we made adjustments to optimize the process, ensuring more time was spent running the training function and less on data movement.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
>
>
> Answer:

[this figure](figures/buckets.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored.**
>
> Answer:

[this figure](figures/registry.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project.**
>
> Answer:

[this figure](figures/build.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- question 23 fill here ---
When writing the API's for our model we considered which method we would need to use. We wanted to be able to train our model in order to make it better. In order to train our model we needed to preprocess pictures in order to have material to train on. lastly we also needed to have an API for sending a picture, preprocess the picture and use the machine to analyze it and return a result whether it was a cat or a dog. To make these API functions we use FastAPI as this seemed like the most intuitive solution. For the preprocess of images and model training we used GET API because we didnt need to send any object now that we have locally put in 30.000 pictures of cats and dogs. For the preprocess we can pass in a number in the url that tells the function how many pictures it needs to preprocess. For the single image evaluation we used a POST function because we need to pass in an image that it needs to evaluate.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---
We tried to deploy our API locally using uvicorn to make a local server where we could call the API using the url. The functions would then get called and would return the training data or some kind of response that the API was sucessfull. It worked perfectly locally and produced the results that we were expecting and it preprocessed and trained on the preprocessed images as intended.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---
For testing of the API we used pytest in order to test the different functions and testClient from the fast api library to simulate a server. we have tested to preprocess data which passed and therefore we can conclude that it works perfectly. We also tested the API for evaluating a single image and it also passed showing that the function works.
We use the patch library from unittest.mock because we would like to the the API function not the other functions inside the API.
In the API where we evaluate a single image by using a patch we create a "dummy" function for the function used inside the API because we do not test the inside function, only the API. By using this patch we ensure that it is only the api we are testing. we assert that the response code is 200 which means that it worked and we also asserts that the "dummy" function is called at least once with the parameters that we send in to the function.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
