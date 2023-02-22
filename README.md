# Simple Neural Net in Go

This code implements a neural network from scratch. It [comes from](https://github.com/dwhitena/gophernet) a walk-through written by [@dwhitena](https://github.com/dwhitena) called "[Building a neural network from scratch in Go](https://datadan.io/blog/neural-net-with-go)". 

After reading some articles about neural nets, including [this article](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) that discusses the mechanics ChatGPT by Stephen Wolfram, I still felt like I had to hand wave away big parts of how neural networks work. This repository is an attempt to pop the hood and get deeper understanding of the systems. And it was really helpful!

If you want to learn the archtecture, training process, and prediction process well, I'd recommend the excercise of writing a neural net from scratch. What this excercise didn't give me was a practical mathematical understanding of neural nets. Walking through the code while reading [this walkthrough](https://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html) helped me understand the math better. (Except I didn't remember how derivatives worked so I'll come update the README if I find any applicable resources on that.)

## Running

From the root of the project run `go run main.go`. 

This command will train and test the network, and then print the accuracy of the model. Each run will result in a different accuracy (usually >90%) because of the random seed to each training session.

## Notes

The training and test data comes from the apparently oft-referenced [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris).

There's some potential directions it might be fun to take this repo:
- [ ] Support multiple hidden layers
- [ ] Print the amount of weights your model configuration will generate
- [ ] Add a diagram to the README that explains how the system works
- [ ] Add unit tests
- [ ] Support other activation functions other than sigmoid
- [ ] Make it much more memory efficient
- [ ] Use the system's GPU for the matrix calculations
- [ ] Generalize the data input code so different datasets can be used
- [ ] Make it into a CLI
- [ ] Build a more sophisticated neural net
- [ ] Build some parallel training
- [ ] Save and load the matrices from previous trainings
- [ ] Figure out how to visualize the trained network
- [ ] Build a benchmark utility

## Contributions

...are welcome! If you see that I got something wrong, have some helpful comments, would like to try the exercise yourself and find a better way of doing it, or want to add some enhancements to the library, feel free to drop a PR.
