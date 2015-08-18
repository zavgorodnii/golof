package lof

import (
    // "fmt"
    "log"
    "math"
)

type LOF struct {
    TrainingSet []ISample
    Distances   [][]DistItem
    KNNs        [][]int
    MinTrain    []float64
    MaxTrain    []float64
    MinPts      int
    NumSamples  int
    AddedIndex  int
}

type DistItem struct {
    Value       float64
    Index       int
}

type LofItem struct {
    Value       float64
    Sample      ISample
}

func NewLOF(minPts int, trnSamples []ISample) *LOF {

    numSamples := len(trnSamples)
    // After training we want to compute LOF values for
    // new samples, and we need some space for their
    // distances; if we find LOF for one new sample at a
    // time, a single additional slot will be enough.
    addedIndex := len(trnSamples) + 1

    if numSamples < minPts {
        log.Fatal("Number of samples is less than MinPts!")
    }

    lof := &LOF{
        TrainingSet: trnSamples,
        MinPts: minPts,
        NumSamples: numSamples,
        AddedIndex: addedIndex,        
    }

    // Prepare storage between training samples
    lof.Distances = make([][]DistItem, addedIndex)
    for idx := 0; idx < addedIndex; idx++ {
        lof.Distances[idx] = make([]DistItem, addedIndex)
    }
    // Prepare storage for each sample's k-neighbors
    lof.KNNs = make([][]int, addedIndex)
    for idx := 0; idx < addedIndex; idx++ {
        lof.KNNs[idx] = make([]int, minPts)
    }

    lof.train(trnSamples)
    return lof
}

func (lof *LOF) train(samples []ISample) {

    // Throughout the function this value  is used for direct indexing
    // (i.e., not inside a for ...;...;... statement), so we need
    // to subtract 1 in order not to get out of range 
    addedIndex := lof.AddedIndex - 1
    numSamples := lof.NumSamples
    for idx, sample := range samples {
        sample.SetId(idx)  // Just additional info
    }

    for i := 0; i < numSamples; i++ {
        // Compute distances between training samples
        for j := 0; j < numSamples; j++ {
            if i == j {
                lof.Distances[i][j].Value = 0  // This is distinctive
                lof.Distances[i][j].Index = j
            } else {
                lof.Distances[i][j].Value = SampleDist(samples[i], samples[j])
                lof.Distances[j][i].Value = lof.Distances[i][j].Value
                lof.Distances[i][j].Index = j
                lof.Distances[j][i].Index = i
            }
        }
        // Set the additional slot's last value
        lof.Distances[addedIndex][addedIndex].Value = 0
        lof.Distances[addedIndex][addedIndex].Index = addedIndex
        lof.updateNNTable(i, "train")
    }
}

func (lof *LOF) GetLOFs(samples []ISample) {

    for _, sample := range samples {
        log.Printf("%v: %f", sample.GetPoint(), lof.GetLOF(sample))
    }
}

func (lof *LOF) GetLOF(added ISample) float64 {

    // Throughout the function this value  is used for direct indexing
    // (i.e., not inside a for ...;...;... statement), so we need
    // to subtract 1 in order not to get out of range 
    addedIndex := lof.AddedIndex - 1
    // Update distances table with added sample
    for i := 0; i < lof.NumSamples; i++ {
        // Distance between current training
        // sample and the sample being added
        dist := DistItem {
            Value: SampleDist(added, lof.TrainingSet[i]),
            Index: addedIndex,
        }
        lof.Distances[i][addedIndex] = dist
        lof.Distances[addedIndex][i] = dist
        lof.Distances[addedIndex][i].Index = i
    }
    // Fill nearest neighbors table for added sample
    // (but don't touch any other samples yet)
    // Find nearest samples for current one: sort distances
    lof.updateNNTable(addedIndex, "compute")

    // Now we want to update nearest neighbors table ONLY
    // for those samples that are the added sample's nearest
    // neighbors; this adds some error, but saves CPU time
    for _, neighborIndex := range lof.KNNs[addedIndex] {
        lof.updateNNTable(neighborIndex, "compute")
    }

    addedDensity := lof.getDensity(addedIndex)
    neighborDensitySum := .0
    for _, neighborIndex := range lof.KNNs[addedIndex] {
        neighborDensitySum += lof.getDensity(neighborIndex)
    }
    factor := (neighborDensitySum / addedDensity) / float64(lof.MinPts)
    return factor
}

// Given a sample's index in Distance table, update this sample's
// row in the nearest neighbors table. The @mode parameter
// controls whether we use the whole table row length (with added
// sample's slots, for LOF computation)
func (lof *LOF) updateNNTable(sampleIndex int, mode string) {

    bound := 0
    switch mode {
    case "train":
        bound = lof.NumSamples
    case "compute":
        bound = lof.AddedIndex
    default:
        log.Fatal("LOF: @mode should be either \"train\" or \"compute\"")
    }
    // Find nearest samples for current one: sort distances
    sorted := make([]DistItem, bound)
    copy(sorted, lof.Distances[sampleIndex])
    SortDistItems(sorted)
    // Find nearest samples for current one: take MinPts nearest
    for k := 1; k < lof.MinPts; k++ {
        lof.KNNs[sampleIndex][k - 1] = sorted[k].Index
    }
}

func (lof *LOF) getDensity(sampleIdx int) float64 {

    distanceSum := .0
    lastNeighborIdx := lof.MinPts - 1
    for _, neighborIdx := range lof.KNNs[sampleIdx] {
        // This is pre-computed distance between target sample
        // and his current nearest neighbors
        distance := lof.Distances[sampleIdx][neighborIdx].Value
        // Index of the farthest sample among the current neigbor's
        // nearest neighbors (will be used to retrieve the actual)
        // distance
        kDistanceIdx := lof.KNNs[neighborIdx][lastNeighborIdx]
        kDistance := lof.Distances[sampleIdx][kDistanceIdx].Value
        distanceSum += math.Max(distance, kDistance)
    }

    return distanceSum / float64(lof.MinPts)
}
