package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	neuralnet "github.com/birdsean/simple-neural-net-go/neural-net"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func readFlowerDataFile(address string) (*mat.Dense, *mat.Dense) {
	trainFile, readErr := os.Open(address)
	if readErr != nil {
		log.Fatal(readErr)
	}
	defer trainFile.Close()

	reader := csv.NewReader(trainFile)
	reader.FieldsPerRecord = 7

	rawCSVData, csvErr := reader.ReadAll()
	if csvErr != nil {
		log.Fatal(csvErr)
	}

	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))
	var inputsIndex int
	var labelsIndex int

	for idx, record := range rawCSVData {
		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns.
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)
	return inputs, labels
}

func main() {
	inputs, labels := readFlowerDataFile("data/train.csv")

	config := neuralnet.NeuralNetConfig{
		CountInputNeurons:  4,
		CountOutputNeurons: 3,
		CountHiddenNeurons: 10,
		CountEpochs:        5000,
		LearningRate:       0.3,
	}

	network := neuralnet.New(config)
	network.Train(inputs, labels)

	testInputs, testLabels := readFlowerDataFile("data/test.csv")
	predictions, err := network.Predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the accuracy of our model.
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Get the label.
		labelRow := mat.Row(nil, i, testLabels)
		var species int
		for idx, label := range labelRow {
			if label == 1.0 {
				species = idx
				break
			}
		}

		// Accumulate the true positive/negative count.
		if predictions.At(i, species) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate the accuracy (subset accuracy).
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Output the Accuracy value to standard out.
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}
