# unicache-env

An [OpenAI-gym](https://github.com/openai/gym) enviroment providing true-world network requesting data

## Install

```
pip install -e .
```

or

```
pip3 install -e .
```

The dataset is too larget to add into this repository. Please refer to `unicache_env/envs/raw/README.md` and put datasets into `unicache_env/envs/raw/`.

## Usage

### Import the environment

You can import different variation of the environment via different Environment ID strings. The format is `"cache-%{datasetString}s-%{param1}s-%{param2}s-v0"`, with each parameter described below. (`"-v0"` is required by *OpenAI Gym* framework)

- `datasetString`: Currently only MovieLens (dataset string = `'movielens'`) dataset is opened.

- `param1`: Storage capacity / total number of contents, must be chosen from `"0.00001"`, `"0.0001"`, `"0.001"`, `"0.01"`.

- `param2`: Sampling size of contents, must be chosen from `"100"`, `"1000"`, `"10000"`, `"full`"(26744 for MovieLens dataset). We provide this parameter because the full set is usually too large, so we have to sample from the dataset.

Pass this string to `gym.make` to make a environment, for example, we choose "cache-movielens-0.0001-full-v0":

```Python
import gym
import unicache_env
env = gym.make("cache-movielens-0.0001-full-v0")
```

### Reset the environment

To reset the environment, run

```Python
state = env.reset()
```

This function returns the initial *state* of the problem. *OpenAI Gym* framework describes problems as automantoms. Every moment, the AI agent is at one state. It observes the current state to make an *action*, which takes it to another state, and gains reward from the action.

A state in this caching problem describes a situation when a new request misses, i.e. the requested content is not cached, but there is no extra caching space available. There are 5 properties in a state:

- `state.storeSize`: An integer. How many contents can be cached in the same time.

- `state.cached`: A NumPy array. `state.cached[i] = true` means Content `i` is cached, and vice versa.

- `state.cachedNum`: An integer. This is a helper property, which equals the number of `ture`s in `state.cached`.

- `state.arriving`: An integer. The Content ID requested by the newly arriving request, which is not cached yet.

- `state.history`:  An array of `Reqest` objects. All the requests received until now. Please refer to its Python doc-string for details.

### Perform an action

Since there is no extra caching space available, and we have to cache the new content requested (We describe the problem as there is no fly-by requests, i.e. all contents should be cached before returning to end users), a content already in cache should be evicted. We refer to this evicting procedure as an *action*.

To perform an action, run

```Python
state, rewards, done, _ = env.step(IDOfTheContentToBeEvicted)
```

This function returns 4 values:

- `state`: A new state the AI agent arrives.

- `reward`: Reward gained from this action. In this problem, it equals to the numbers of requests that *hits*, i.e. the requested content is already cached, after leaving the old state and arriving at the new state.

- `done`: This is a boolean value, true for all the requests have been handled.

- `_`: This field is reserved for returning extra info in the future.
