package main

import (
	"errors"
	"fmt"
	"math/rand"

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

type Agent struct {
	gamma  float64
	policy Policy
	memory []Memory
	cnt    map[[2]int]int
	v      map[[2]int]float64
}

type Memory struct {
	state  [2]int
	action int
	reward float64
}

func newAgent(gamma float64, policy Policy, states [][2]int) *Agent {
	cnt := make(map[[2]int]int)
	v := make(map[[2]int]float64)
	for _, state := range states {
		cnt[state] = 0
		v[state] = 0.0
	}

	return &Agent{
		gamma:  gamma,
		policy: policy,
		memory: make([]Memory, 0),
		cnt:    cnt,
		v:      v,
	}
}

func (a *Agent) addMemory(state [2]int, action int, reward float64) {
	a.memory = append(a.memory, Memory{state, action, reward})
}
func (a *Agent) getAction(state [2]int) (int, error) {
	statePolicy := a.policy[state]
	cumProb := 0.0
	for action, prob := range statePolicy {
		cumProb += prob
		if rand.Float64() < cumProb {
			return action, nil
		}
	}
	return -1, errors.New("action not found")
}

func (a *Agent) step(state [2]int, action int, m *maze.Maze) ([2]int, float64, bool) {
	goalX, goalY, err := m.GetGoal()
	if err != nil {
		panic(err)
	}
	nextState := getNextState(state, action, m)
	reward := getReward(nextState, goalX, goalY)
	isGoal := nextState[0] == goalX && nextState[1] == goalY
	return nextState, reward, isGoal

}

func (a *Agent) eval(goalX, goalY int) {
	g := 0.0
	for i := len(a.memory) - 1; i >= 0; i-- {
		memory := a.memory[i]
		if memory.state[0] == goalX && memory.state[1] == goalY {
			a.v[memory.state] = 0.0
			continue
		}
		g = a.gamma*g + memory.reward
		a.cnt[memory.state]++
		a.v[memory.state] += (g - a.v[memory.state]) / float64(a.cnt[memory.state])
	}
}

func (a *Agent) reset() {
	a.memory = make([]Memory, 0)
}

func newPolicy(m *maze.Maze) Policy {
	states := getStates(m)
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

func getNextState(state [2]int, action int, m *maze.Maze) [2]int {
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
	if m.IsAvailable(nextStateCandidate[0], nextStateCandidate[1]) {
		return nextStateCandidate

	}
	return state

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

func iterEpisodes(episodes int, agent *Agent, dungeon *maze.Maze) {
	states := getStates(dungeon)
	goalX, goalY, err := dungeon.GetGoal()
	if err != nil {
		panic(err)
	}
	for i := 0; i < episodes; i++ {
		state := states[0]
		agent.reset()
		for {
			action, err := agent.getAction(state)
			if err != nil {
				panic(err)
			}
			nextState, reward, isGoal := agent.step(state, action, dungeon)
			agent.addMemory(state, action, reward)
			if isGoal {
				agent.eval(goalX, goalY)
				break
			}
			state = nextState
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
	policy := newPolicy(dungeon)
	states := getStates(dungeon)
	agent := newAgent(0.9, policy, states)
	iterEpisodes(1000, agent, dungeon)
	fmt.Println(agent.v)
}
