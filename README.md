<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project was part of an AI course I took and it had us build a Multi-Layer Perceptron (MLP) neural network with a backpropagation method of training. A single MLP is constructed with one hidden layer and one output layer. Using the a logistic activation function for calculating the gradient along with a user defined learning rate for the backpropagation. We were tasked with not using any outside ANN/MLP libraries/code for the project and in a sense doing it “from scratch”.

![MLP Architecture][mlp-architecture]*(Source: https://blog.goodaudience.com/artificial-neural-networks-explained-436fcf36e75)*

First this MLP is designed and structured to solve the XOR problem. The feedforward algorithm takes the values from the input nodes which can either `0` or `1` and a bias value of `1` and multiplys them with its corresponding weights to get nodes for the hidden layer. In this case I only implement one hidden layer, but the addition of other hidden layers would keep propagating forward until the output layer is reached. Next each hidden layer value not including the bias invokes an activation function, and for this project I use the sigmoid function which puts the sum of their input values to fall between `0` and `1`.
<p align="center">
	<img width="250" src="https://github.com/ocastaneda3/multilayer-perceptron/blob/main/images/sigmoid.png">
</p>

The outputs of each hidden layer unit, including the bias unit, are then multiplied again but now with their respective weights to get the output layer value(s). The output value(s) also go through the activation function and returns a value falling between 0 and 1. This is the predicted output.

To train the MLP I implement a backpropagation algorithm in order a good set of weight values to be found to make the best guess. This is done by first comparing the output value(s) obtained from forward propagation and the expected output value(s) to calculate error values. With these errors we start moving backwards through the network calculating some delta values. These delta values calculated are slight tweaks that move the wights in a direction that reduces the size of the error by a small degree. Here another activation function is applied, derivative of the sigmoid function. This is because while backpropagating during the training of the MLP we need to find the derivative of the loss function with respect to each weight in the network. And since the activation function is the sigmoid funtion, the derivative of it is used.
<p align="center">
	<img width="250" src="https://miro.medium.com/0*s-oj85y4gHExvkx0">
</p>


### Built With

* [Python 3.9.0](https://www.python.org/downloads/release/python-390/)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
2. Install Python packages
   ```sh
   pip3 install -r requirements.txt
   ```

### Usage
1. Run
  ```sh
  python3 mlp.py
  ```
  
  <p align="center">
	  <img src="https://github.com/ocastaneda3/multilayer-perceptron/blob/main/images/output.png">
  </p>

<!-- CONTACT -->
## Contact

Project Link: [https://github.com/ocastaneda3/multilayer-perceptron](https://github.com/ocastaneda3/multilayer-perceptron)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ocastaneda3/multilayer-perceptron.svg?style=for-the-badge
[contributors-url]: https://github.com/ocastaneda3/multilayer-perceptron/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ocastaneda3/multilayer-perceptron.svg?style=for-the-badge
[forks-url]: https://github.com/ocastaneda3/multilayer-perceptron/network/members
[stars-shield]: https://img.shields.io/github/stars/ocastaneda3/multilayer-perceptron.svg?style=for-the-badge
[stars-url]: https://github.com/ocastaneda3/multilayer-perceptron/stargazers
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/oscar-castaneda93/

[mlp-architecture]: images/mlp_architecture.png
[sigmoid-func]: images/sigmoid.PNG
[sigmoid-derivative-func]: images/sigmoid_derivative.PNG
