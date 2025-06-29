import numpy as np
import random
from typing import Dict, List, Tuple, Any
from collections import deque
from optimizers.BaseOptimizer import BaseOptimizer


class SimpleNeuralNetwork:
    """Simple neural network for Q-function approximation"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.001,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        """Forward pass through network"""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Hidden layer with ReLU activation
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU

        # Output layer (linear)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def train_step(self, x, target):
        """Single training step using gradient descent"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)

        # Forward pass
        output = self.forward(x)

        # Backward pass
        # Output layer gradients
        dZ2 = output - target
        dW2 = np.dot(self.a1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.z1 > 0)  # ReLU derivative
        dW1 = np.dot(x.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        return np.mean((output - target) ** 2)  # Return MSE loss


class PureExplorationOptimizer(BaseOptimizer):
    """
    Pure Exploration DQN - Uses Q-network only for intelligent action selection,
    but always explores (never exploits). Tests the hypothesis that exploitation
    is unnecessary for hyperparameter optimization.
    """

    def __init__(
        self,
        hyperparameter_space: Dict,
        learning_rate: float = 0.001,
        hidden_size: int = 64,
        memory_size: int = 1000,
        batch_size: int = 32,
    ):
        """
        Initialize Pure Exploration Optimizer

        Args:
            hyperparameter_space: Dictionary defining the hyperparameter search space
            learning_rate: Learning rate for neural network
            hidden_size: Hidden layer size for Q-network
            memory_size: Size of experience replay buffer
            batch_size: Batch size for network training
        """
        super().__init__("PureExploration", memory_size)

        self.hyperparameter_space = hyperparameter_space
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Create action space (same as DQN)
        self.actions = self._create_action_space()
        self.num_actions = len(self.actions)

        # Create state space
        self.state_size = self._calculate_state_size()

        # Initialize Q-network (for intelligent action selection)
        self.q_network = SimpleNeuralNetwork(
            input_size=self.state_size,
            hidden_size=hidden_size,
            output_size=self.num_actions,
            learning_rate=learning_rate,
        )

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)

        # Track optimization state
        self.current_params = None
        self.param_history = []
        self.score_history = []
        self.rounds_since_improvement = 0
        self.exploration_counts = {}

        # Initialize parameter ranges for normalization
        self.param_ranges = self._get_parameter_ranges()

        # Strategy counters for analysis
        self.strategy_counts = {
            "random": 0,
            "around_best": 0,
            "extremes": 0,
            "network_guided": 0,
        }

        print(
            f"PureExploration: ALWAYS explores, {self.num_actions} actions, state_size={self.state_size}"
        )
        print(
            f"Strategy: Q-network guides action selection, but never exploits same action"
        )

    def _create_action_space(self) -> List[Dict]:
        """Create discrete action space for parameter modifications"""
        actions = []

        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] in ["int", "float"]:
                # Fine-grained parameter adjustments
                actions.extend(
                    [
                        # Tiny adjustments for fine-tuning
                        {
                            "name": f"increase_{param_name}_tiny",
                            "param": param_name,
                            "type": "multiply",
                            "factor": 1.05,
                        },
                        {
                            "name": f"decrease_{param_name}_tiny",
                            "param": param_name,
                            "type": "multiply",
                            "factor": 0.95,
                        },
                        # Small adjustments
                        {
                            "name": f"increase_{param_name}_small",
                            "param": param_name,
                            "type": "multiply",
                            "factor": 1.15,
                        },
                        {
                            "name": f"decrease_{param_name}_small",
                            "param": param_name,
                            "type": "multiply",
                            "factor": 0.85,
                        },
                        # Medium adjustments
                        {
                            "name": f"increase_{param_name}_medium",
                            "param": param_name,
                            "type": "multiply",
                            "factor": 1.3,
                        },
                        {
                            "name": f"decrease_{param_name}_medium",
                            "param": param_name,
                            "type": "multiply",
                            "factor": 0.7,
                        },
                        # Large adjustments
                        {
                            "name": f"increase_{param_name}_large",
                            "param": param_name,
                            "type": "multiply",
                            "factor": 2.0,
                        },
                        {
                            "name": f"decrease_{param_name}_large",
                            "param": param_name,
                            "type": "multiply",
                            "factor": 0.5,
                        },
                    ]
                )
            elif param_config["type"] == "categorical":
                # For categorical parameters, create switch actions
                for value in param_config["values"]:
                    actions.append(
                        {
                            "name": f"set_{param_name}_to_{value}",
                            "param": param_name,
                            "type": "set",
                            "value": value,
                        }
                    )

        # Add exploration actions
        actions.extend(
            [
                {
                    "name": "explore_new_region_random",
                    "type": "explore",
                    "strategy": "random",
                },
                {
                    "name": "explore_around_best",
                    "type": "explore",
                    "strategy": "around_best",
                },
                {
                    "name": "explore_parameter_extremes",
                    "type": "explore",
                    "strategy": "extremes",
                },
            ]
        )

        return actions

    def _calculate_state_size(self) -> int:
        """Calculate the size of the state vector"""
        state_size = 0
        state_size += 1  # current_best_score
        state_size += 1  # rounds_since_improvement
        state_size += 1  # progress (current_round / max_rounds)
        state_size += len(self.hyperparameter_space)  # current_params_normalized
        state_size += 1  # recent_performance_trend
        state_size += len(
            self.hyperparameter_space
        )  # exploration_density per parameter
        return state_size

    def _get_parameter_ranges(self) -> Dict:
        """Get parameter ranges for normalization"""
        ranges = {}
        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] == "int":
                ranges[param_name] = (param_config["min"], param_config["max"])
            elif param_config["type"] == "float":
                ranges[param_name] = (param_config["min"], param_config["max"])
            elif param_config["type"] == "categorical":
                ranges[param_name] = (0, len(param_config["values"]) - 1)
        return ranges

    def _normalize_parameters(self, params: Dict) -> np.ndarray:
        """Normalize parameters to [0, 1] range for state representation"""
        normalized = []
        for param_name in sorted(self.hyperparameter_space.keys()):
            param_config = self.hyperparameter_space[param_name]
            value = params.get(param_name, 0)

            if param_config["type"] in ["int", "float"]:
                min_val, max_val = self.param_ranges[param_name]
                normalized_val = (
                    (value - min_val) / (max_val - min_val)
                    if max_val > min_val
                    else 0.5
                )
                normalized.append(np.clip(normalized_val, 0, 1))
            elif param_config["type"] == "categorical":
                try:
                    idx = param_config["values"].index(value)
                    normalized_val = (
                        idx / (len(param_config["values"]) - 1)
                        if len(param_config["values"]) > 1
                        else 0.5
                    )
                    normalized.append(normalized_val)
                except ValueError:
                    normalized.append(0.5)

        return np.array(normalized)

    def _get_current_state(self, round_num: int) -> np.ndarray:
        """Get current state representation"""
        state = []

        # Current best score (normalized)
        if len(self.score_history) > 0:
            max_possible_score = 1.0
            current_best_normalized = self.best_score / max_possible_score
        else:
            current_best_normalized = 0.0
        state.append(current_best_normalized)

        # Rounds since improvement (normalized)
        max_rounds_without_improvement = 20
        rounds_since_norm = (
            min(self.rounds_since_improvement, max_rounds_without_improvement)
            / max_rounds_without_improvement
        )
        state.append(rounds_since_norm)

        # Progress through optimization
        max_rounds = 500
        progress = min(round_num, max_rounds) / max_rounds
        state.append(progress)

        # Current parameters (normalized)
        if self.current_params is not None:
            normalized_params = self._normalize_parameters(self.current_params)
        else:
            normalized_params = np.full(len(self.hyperparameter_space), 0.5)
        state.extend(normalized_params)

        # Recent performance trend
        if len(self.score_history) >= 3:
            recent_scores = self.score_history[-3:]
            trend = (recent_scores[-1] - recent_scores[0]) / 2
            state.append(np.clip(trend, -1, 1))
        else:
            state.append(0.0)

        # Exploration density per parameter
        for param_name in sorted(self.hyperparameter_space.keys()):
            exploration_count = self.exploration_counts.get(param_name, 0)
            max_exploration = max(10, round_num)
            exploration_density = (
                min(exploration_count, max_exploration) / max_exploration
            )
            state.append(exploration_density)

        return np.array(state, dtype=np.float32)

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggest next hyperparameters using Pure Exploration with Q-guided action selection"""

        # Initialize with random parameters if first round
        if self.current_params is None:
            self.current_params = self._generate_random_parameters()
            print(f"Round {round_num}: INITIALIZATION")
            print(f"Params: {self.current_params}")
            return self.current_params

        # Get current state
        state = self._get_current_state(round_num)

        # PURE EXPLORATION: Use Q-network to rank actions, then sample from top actions
        q_values = self.q_network.forward(state)[0]

        # Get top 30% of actions (exploration from good actions)
        top_k = max(3, int(0.3 * self.num_actions))
        top_action_indices = np.argsort(q_values)[-top_k:]

        # Randomly select from top actions (weighted by Q-value)
        q_scores = q_values[top_action_indices]
        q_scores_normalized = np.exp(q_scores - np.max(q_scores))  # Softmax-like
        probabilities = q_scores_normalized / np.sum(q_scores_normalized)

        action_idx = np.random.choice(top_action_indices, p=probabilities)
        selected_action = self.actions[action_idx]

        # Apply selected action
        new_params = self._apply_action(selected_action, self.current_params)

        # Store for next update
        self.last_state = state
        self.last_action_idx = action_idx
        self.last_action = selected_action

        # Update current parameters
        self.current_params = new_params

        # Update exploration counts
        for param_name in new_params.keys():
            self.exploration_counts[param_name] = (
                self.exploration_counts.get(param_name, 0) + 1
            )

        # Track strategy usage
        if selected_action["type"] == "explore":
            self.strategy_counts[selected_action["strategy"]] += 1
        else:
            self.strategy_counts["network_guided"] += 1

        print(f"Round {round_num}: PURE_EXPLORATION (Q-guided)")
        print(
            f"Action: {selected_action['name']} (Q-value: {q_values[action_idx]:.3f})"
        )
        print(f"Strategy counts: {self.strategy_counts}")
        print(f"Params: {new_params}")

        return new_params

    def _apply_action(self, action: Dict, current_params: Dict) -> Dict:
        """Apply an action to current parameters"""
        new_params = current_params.copy()

        if action["type"] == "multiply":
            param_name = action["param"]
            param_config = self.hyperparameter_space[param_name]
            current_value = current_params[param_name]
            new_value = current_value * action["factor"]

            # Clip to parameter bounds
            if param_config["type"] == "int":
                new_value = int(
                    np.clip(new_value, param_config["min"], param_config["max"])
                )
            elif param_config["type"] == "float":
                new_value = np.clip(new_value, param_config["min"], param_config["max"])

            new_params[param_name] = new_value

        elif action["type"] == "set":
            param_name = action["param"]
            new_params[param_name] = action["value"]

        elif action["type"] == "explore":
            strategy = action.get("strategy", "random")

            if strategy == "random":
                new_params = self._generate_random_parameters()
            elif strategy == "around_best" and self.best_params is not None:
                new_params = self._explore_around_best()
            elif strategy == "extremes":
                new_params = self._explore_extremes(current_params)
            else:
                new_params = self._generate_random_parameters()

        return new_params

    def _generate_random_parameters(self) -> Dict:
        """Generate random parameters within the search space"""
        params = {}
        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] == "int":
                params[param_name] = random.randint(
                    param_config["min"], param_config["max"]
                )
            elif param_config["type"] == "float":
                params[param_name] = random.uniform(
                    param_config["min"], param_config["max"]
                )
            elif param_config["type"] == "categorical":
                params[param_name] = random.choice(param_config["values"])
        return params

    def _explore_around_best(self) -> Dict:
        """Explore parameters around the current best known parameters"""
        if self.best_params is None:
            return self._generate_random_parameters()

        new_params = self.best_params.copy()

        # Randomly modify 1-3 parameters around best values
        num_params_to_modify = random.randint(1, min(3, len(new_params)))
        params_to_modify = random.sample(list(new_params.keys()), num_params_to_modify)

        for param_name in params_to_modify:
            param_config = self.hyperparameter_space[param_name]
            current_value = new_params[param_name]

            if param_config["type"] in ["int", "float"]:
                # Add small random perturbation (±20%)
                perturbation = random.uniform(0.8, 1.2)
                new_value = current_value * perturbation

                # Clip to bounds
                if param_config["type"] == "int":
                    new_value = int(
                        np.clip(new_value, param_config["min"], param_config["max"])
                    )
                elif param_config["type"] == "float":
                    new_value = np.clip(
                        new_value, param_config["min"], param_config["max"]
                    )

                new_params[param_name] = new_value

            elif param_config["type"] == "categorical":
                if random.random() < 0.3:
                    new_params[param_name] = random.choice(param_config["values"])

        return new_params

    def _explore_extremes(self, current_params: Dict) -> Dict:
        """Explore extreme parameter values"""
        new_params = current_params.copy()

        # Pick 1-2 parameters to set to extreme values
        num_params_to_modify = random.randint(1, min(2, len(new_params)))
        params_to_modify = random.sample(list(new_params.keys()), num_params_to_modify)

        for param_name in params_to_modify:
            param_config = self.hyperparameter_space[param_name]

            if param_config["type"] in ["int", "float"]:
                if random.random() < 0.5:
                    new_params[param_name] = param_config["min"]
                else:
                    new_params[param_name] = param_config["max"]
            elif param_config["type"] == "categorical":
                new_params[param_name] = random.choice(param_config["values"])

        return new_params

    def update(self, hyperparameters: Dict, score: float):
        """Update Q-network with the result of the last suggestion"""

        # Update score history
        self.score_history.append(score)

        # Calculate reward
        if len(self.score_history) == 1:
            reward = score
            improvement = True
        else:
            previous_best = max(self.score_history[:-1])
            if score > previous_best:
                improvement_magnitude = score - previous_best
                reward = improvement_magnitude * 10
                self.rounds_since_improvement = 0
                improvement = True
            else:
                reward = -0.01
                self.rounds_since_improvement += 1
                improvement = False

        # Update global best tracking
        old_best = self.best_score
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters.copy()
            print(f"  *** NEW BEST SCORE: {score:.4f} (was {old_best:.4f}) ***")
        else:
            print(f"  No improvement: {score:.4f} vs best {self.best_score:.4f}")

        print(
            f"Reward: {reward:.4f}, Rounds since improvement: {self.rounds_since_improvement}"
        )

        # Store experience in replay buffer
        if hasattr(self, "last_state") and hasattr(self, "last_action_idx"):
            current_state = self._get_current_state(len(self.score_history))
            experience = (
                self.last_state,
                self.last_action_idx,
                reward,
                current_state,
                False,
            )
            self.memory.append(experience)

            # Train Q-network
            if len(self.memory) >= self.batch_size:
                self._train_q_network()

        # Add to history
        self.history.append({"params": hyperparameters, "score": score})

    def _train_q_network(self):
        """Train Q-network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)

        # Prepare training data
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])

        # Current Q-values
        current_q_values = self.q_network.forward(states)

        # Next Q-values
        next_q_values = self.q_network.forward(next_states)

        # Calculate targets
        gamma = 0.95
        targets = current_q_values.copy()

        for i in range(self.batch_size):
            target = rewards[i] + gamma * np.max(next_q_values[i])
            targets[i, actions[i]] = target

        # Train network
        total_loss = 0
        for i in range(self.batch_size):
            loss = self.q_network.train_step(states[i : i + 1], targets[i : i + 1])
            total_loss += loss

        avg_loss = total_loss / self.batch_size
        if len(self.history) % 20 == 0:
            print(f"  Q-network training loss: {avg_loss:.6f}")
