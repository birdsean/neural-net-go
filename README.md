# Simple Neural Net in Go

This code implements a neural network from scratch. It was [heavily inspired](https://github.com/dwhitena/gophernet) by a walk-through written by [@dwhitena](https://github.com/dwhitena) called "[Building a neural network from scratch in Go](https://datadan.io/blog/neural-net-with-go)". 

After reading some articles about neural nets, including [this article](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) that discusses the mechanics ChatGPT by Stephen Wolfram, I still felt like I had to hand wave away big parts of how neural networks work. This repository is an attempt to pop the hood and get deeper understanding of the systems. And it was really helpful!

If you want to learn the archtecture, training process, and prediction process well, I'd recommend the excercise of writing a neural net from scratch. What this excercise didn't give me was the practical mathematical understanding of neural nets that I think is needed. So I guess I'll have to go get that somewhere else. I'll come back and update this README if I find a good way to do that.

## Running

From the root of the project run `go run main.go`. 

This command will train and test the network, and then print the accuracy of the model. Each run will result in a different accuracy (usually >90%) because of the random seed to each training session.

## Notes

The training and test data comes from the apparently oft-referenced [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris).

There's some potential directions it might be fun to take this repo:
- [ ] Generalize the data input code so different datasets can be used
- [ ] Support other activation functions other than sigmoid
- [ ] Make it much more memory efficient
- [ ] Use the system's GPU for the matrix calculations
- [ ] Add a diagram to the README that explains how the system works
- [ ] Make it into a CLI
- [ ] Build a more sophisticated neural net
- [ ] Add unit tests
- [ ] Save and load the matrices from previous trainings

## Contributions

...are welcome! If you see that I got something wrong, have some helpful comments, would like to try the exercise yourself, or want to add some enhancments to the library, feel free to drop a PR.
