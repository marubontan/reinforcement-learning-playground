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

type HistoryElement struct {
	state  [2]int
	action int
	reward float64
	isGoal bool
}

var actions = []int{Left, Right, Up, Down}

type Agent struct {
	gamma   float64
	policy  Policy
	alpha   float64
	epsilon float64
	q       map[[2]int]map[int]float64
	memory  [2]*HistoryElement
}

func newAgent(gamma float64, alpha float64, epsilon float64, policy Policy, states [][2]int, actions []int) *Agent {
	q := make(map[[2]int]map[int]float64)
	for _, state := range states {
		q[state] = make(map[int]float64)
		for _, action := range actions {
			q[state][action] = 0.0
		}
	}
	memory := [2]*HistoryElement{nil, nil}

	return &Agent{
		gamma:   gamma,
		policy:  policy,
		epsilon: epsilon,
		alpha:   alpha,
		q:       q,
		memory:  memory,
	}
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

func (a *Agent) reset() {
	a.memory[0] = nil
	a.memory[1] = nil
}

func (a *Agent) update(state [2]int, action int, reward float64, isGoal bool) {
	a.memory[1] = a.memory[0]
	a.memory[0] = &HistoryElement{
		state:  state,
		action: action,
		reward: reward,
		isGoal: isGoal,
	}
	if a.memory[1] == nil {
		return
	}
	var nextQ float64
	if isGoal {
		nextQ = 0.0
	} else {
		nextQ = a.q[a.memory[1].state][a.memory[1].action]
	}
	target := reward + a.gamma*nextQ
	a.q[state][action] += (target - a.q[state][action]) * a.alpha

	a.policy[state] = a.greedyProbs(state)
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
func argmax(data map[int]float64) int {
	var maxKey int
	var maxValue float64

	for key, value := range data {
		maxKey = key
		maxValue = value
		break
	}

	for key, value := range data {
		if value > maxValue {
			maxKey = key
			maxValue = value
		}
	}

	return maxKey
}
func (a *Agent) greedyProbs(state [2]int) map[int]float64 {
	stateQ := a.q[state]
	maxAction := argmax(stateQ)
	baseProb := a.epsilon / float64(len(actions))
	actionProbs := make(map[int]float64)
	for _, action := range actions {
		if action == maxAction {
			actionProbs[action] = 1.0 - a.epsilon + baseProb
		} else {
			actionProbs[action] = baseProb
		}
	}
	return actionProbs
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
	for i := 0; i < episodes; i++ {
		state := states[0]
		agent.reset()
		for {
			action, err := agent.getAction(state)
			if err != nil {
				panic(err)
			}
			nextState, reward, isGoal := agent.step(state, action, dungeon)
			agent.update(state, action, reward, isGoal)
			if isGoal {
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
	agent := newAgent(0.9, 0.5, 0.1, policy, states, actions)
	iterEpisodes(10000, agent, dungeon)
	fmt.Println(agent.q)
}
