package main

import (
	"fmt"
	"math"

	collections "github.com/marubontan/go-collections"
	"github.com/marubontan/go-maze/maze"
)

func composeDungen() *maze.Maze {
	dungeon := maze.NewMaze(3, 4)
	var err error
	err = dungeon.SetStart(0, 2)
	if err != nil {
		panic(err)
	}
	err = dungeon.SetGoal(3, 0)
	if err != nil {
		panic(err)
	}
	err = dungeon.SetObstacle(1, 1)
	if err != nil {
		panic(err)
	}
	return dungeon
}

func printDungenConf() {
	fmt.Println("Dungeon Configuration:")
	fmt.Println("S: Start Position")
	fmt.Println("X: Obstacle")
	fmt.Println(("G: Goal with Reward 1"))
}

const (
	Left = iota
	Right
	Up
	Down
)

func newV() *collections.DefaultDict[[2]int, float64] {
	v := collections.NewDefaultDict[[2]int, float64]()
	return &v
}

type State [][2]int

func getStates(m *maze.Maze) State {
	blockIndices := make(State, 0)
	for hI, hBlocks := range m.Blocks {
		for wI := range hBlocks {
			blockIndices = append(blockIndices, [2]int{wI, hI})
		}
	}
	return blockIndices
}

type Policy map[[2]int]map[int]float64

var actions = []int{Left, Right, Up, Down}

func newPolicy(states [][2]int, m *maze.Maze) Policy {
	var policy Policy = make(map[[2]int]map[int]float64)
	for _, state := range states {
		statePolicy := make(map[int]float64)
		for _, action := range actions {
			statePolicy[action] = 0.25
		}
		policy[state] = statePolicy
	}
	return policy

}

func getNextState(prevState [2]int, newStateCandidate [2]int, m *maze.Maze) [2]int {
	if m.IsAvailable(newStateCandidate[0], newStateCandidate[1]) {
		return newStateCandidate

	}
	return prevState

}

func getReward(state [2]int, goalX, goalY int) float64 {
	if state[0] == goalX && state[1] == goalY {
		return 1.0
	}
	if state[0] == 3 && state[1] == 1 {
		return -1.0
	}
	return 0
}

func iterValueOneStep(policy Policy, v *collections.DefaultDict[[2]int, float64], dungeon *maze.Maze, gamma float64) *collections.DefaultDict[[2]int, float64] {
	states := getStates(dungeon)
	goalX, goalY, err := dungeon.GetGoal()
	if err != nil {
		panic(err)
	}

	for _, state := range states {
		if state[0] == goalX && state[1] == goalY {
			v.Set(state, 0.0)
			continue
		}
		statePolicy := policy[state]
		actionValues := make([]float64, 0)
		for action := range statePolicy {
			var nextStateCandidate [2]int
			nextStateCandidate[0] = state[0]
			nextStateCandidate[1] = state[1]
			switch action {
			case Left:
				nextStateCandidate[0]--
			case Right:
				nextStateCandidate[0]++
			case Up:
				nextStateCandidate[1]--
			case Down:
				nextStateCandidate[1]++
			}
			nextState := getNextState(state, nextStateCandidate, dungeon)
			reward := getReward(nextState, goalX, goalY)
			actionValues = append(actionValues, (reward + gamma*v.Get(nextState)))
		}
		v.Set(state, findMaxValue((actionValues)))

	}
	return v

}

func findMaxValue(arr []float64) float64 {
	if len(arr) == 0 {
		return 0
	}

	maxValue := arr[0]

	for _, value := range arr[1:] {
		if value > maxValue {
			maxValue = value
		}
	}

	return maxValue
}

func iterValue(policy Policy, v *collections.DefaultDict[[2]int, float64], dungeon *maze.Maze, gamma float64) {
	for {
		oldV := collections.NewDefaultDict[[2]int, float64]()
		for state, value := range v.Data {
			oldV.Set(state, value)
		}
		iterValueOneStep(policy, v, dungeon, gamma)
		var delta float64 = -1
		for state := range v.Data {
			if presentDelta := math.Abs(oldV.Get(state) - v.Get(state)); presentDelta > delta {
				delta = presentDelta
			}
		}
		if delta == 0 {
			return
		}
	}
}

func main() {
	dungeon := composeDungen()
	fmt.Println("=========================================")
	fmt.Println("Dungeon:")
	dungeon.Print()
	fmt.Println("=========================================")
	printDungenConf()
	fmt.Println("=========================================")
	v := newV()
	policy := newPolicy(getStates(dungeon), dungeon)
	iterValue(policy, v, dungeon, 0.9)
	fmt.Println(v)
}
