package main

import (
	"fmt"

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
			if m.Blocks[hI][wI].BlockType != maze.Obstacle {
				blockIndices = append(blockIndices, [2]int{wI, hI})
			}
		}
	}
	return blockIndices
}

type Policy map[[2]int]map[int]float64

var actions = []int{Left, Right, Up, Down}

func nextBlockAvailable(x, y, action int, m *maze.Maze) bool {
	nextX := x
	nextY := y
	switch action {
	case Left:
		nextX--
	case Right:
		nextX++
	case Up:
		nextY--
	case Down:
		nextY++
	}
	return m.IsAvailable(nextX, nextY)

}

func newPolicy(states [][2]int, m *maze.Maze) Policy {
	var policy Policy = make(map[[2]int]map[int]float64)
	for _, state := range states {
		statePolicy := make(map[int]float64)
		availableActions := make([]int, 0)
		for _, action := range actions {
			if nextBlockAvailable(state[0], state[1], action, m) {
				availableActions = append(availableActions, action)
			}
		}
		for _, action := range availableActions {
			statePolicy[action] = 1.0 / float64(len(availableActions))
		}
		policy[state] = statePolicy
	}
	return policy

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

func evalStep(policy Policy, v *collections.DefaultDict[[2]int, float64], dungeon *maze.Maze, gamma float64) *collections.DefaultDict[[2]int, float64] {
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
		var newV float64
		for action, prob := range statePolicy {
			var nextState [2]int
			nextState[0] = state[0]
			nextState[1] = state[1]
			switch action {
			case Left:
				nextState[0]--
			case Right:
				nextState[0]++
			case Up:
				nextState[1]--
			case Down:
				nextState[1]++
			}
			reward := getReward(nextState, goalX, goalY)
			newV += prob * (reward + gamma*v.Get(nextState))
		}
		v.Set(state, newV)

	}
	return v

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
	for i := 0; i < 100; i++ {
		evalStep(policy, v, dungeon, 0.9)
	}
	fmt.Println(v)
}
