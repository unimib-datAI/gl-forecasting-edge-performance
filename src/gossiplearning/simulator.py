import sys
from queue import PriorityQueue

import numpy as np

from gossiplearning.aggregators import choose_aggregator
from gossiplearning.config import Config
from gossiplearning.event import (
    SendModelsLoopEvent,
    process_event,
)
from gossiplearning.history import (
    History,
)
from gossiplearning.log import Logger
from gossiplearning.models import (
    ModelBuilder,
    AggregatorFn,
    LabelledData,
    NodeWeightFn,
)
from gossiplearning.models import NodeDataFn
from gossiplearning.node import Node
from gossiplearning.weight import weight_by_dataset_size
from gossiplearning.weights_marshaling import (
    MarshalWeightsFn,
    marshal_weights_with_random_subsampling,
)


class Simulator:
    """
    Gossip learning simulator runner.
    """

    def __init__(
        self,
        *,
        create_model: ModelBuilder,
        config: Config,
        node_data_fn: NodeDataFn,
        test_set: LabelledData,
        aggregator_fn: AggregatorFn | None = None,
        weight_fn: NodeWeightFn = weight_by_dataset_size,
        marshal_weights_fn: MarshalWeightsFn = marshal_weights_with_random_subsampling,
    ):
        """
        Initialize the simulator state.

        :param create_model: the function used to create models
        :param config: the simulator config
        """
        Config.model_validate(config)
        self._config = config

        self._history: History = History()

        self._events_queue = PriorityQueue(maxsize=0)

        self._logger = Logger(
            log_level=config.log_level, workspace_dir=config.workspace_dir
        )

        aggregator = choose_aggregator(config.training.merge_strategy, aggregator_fn)

        self._nodes = tuple(
            Node(
                id=node_conf.id,
                links=node_conf.links,
                training_config=config.training,
                create_model_fn=create_model,
                workspace_dir=config.workspace_dir,
                logger=self._logger,
                node_data_fn=node_data_fn,
                aggregator=aggregator,
                marshal_weights_fn=marshal_weights_fn,
                test_set=test_set,
                history_config=config.history,
                weight_fn=weight_fn,
            )
            for node_conf in config.nodes
        )

    def run_training_simulation(self) -> History:
        """
        Run the whole gossip learning simulation.

        Events are processed sequentially following their (discrete) time, which is measured in
        seconds since the beginning of the simulation.

        :return: the history object containing all events
        """
        for i in range(len(self._nodes)):
            time = np.random.choice(np.arange(60))

            self._events_queue.put(SendModelsLoopEvent(time=time, handler_node_id=i))

        num_stopped_nodes = 0
        while (
            not self._events_queue.empty() and num_stopped_nodes < self._config.n_nodes
        ):
            event = self._events_queue.get()
            new_events = process_event(
                event,
                node=self._nodes[event.handler_node_id],
                logger=self._logger,
                history=self._history,
                training_config=self._config.training,
            )

            for new_event in new_events:
                self._events_queue.put(new_event)

            if len(self._history.stopped_time) > num_stopped_nodes:
                num_stopped_nodes = len(self._history.stopped_time)
                print(
                    f"{num_stopped_nodes}/{self._config.n_nodes} stopped! (Time {event.time}s)"
                )
                sys.stdout.flush()

        for node in self._nodes:
            if self._config.training.finetuning_epochs > 0:
                _, best_weights, best_val_loss = node.train_model(
                    n_epochs=self._config.training.finetuning_epochs
                )
                node.update_best_model(
                    weights=best_weights,
                    val_loss=best_val_loss,
                )
            node.persist_best_model()

        self._history.nodes_training_history = {
            node.id: node.training_history for node in self._nodes
        }
        self._history.nodes_test_history = {
            node.id: {
                "mse": [m.mse for m in node.eval_metrics],
                "rmse": [m.rmse for m in node.eval_metrics],
                "msle": [m.msle for m in node.eval_metrics],
                "mae": [m.mae for m in node.eval_metrics],
            }
            for node in self._nodes
        }

        return self._history
