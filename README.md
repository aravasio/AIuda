# catalog

Paste a Hugging Face model link. Find out what the model does, and whether your
machine can run it.

```
catalog query https://huggingface.co/Qwen/QwQ-32B
```

Everything is worked out from the repo itself. There is no hand-written catalog
to keep up to date, so a model published tomorrow reads the same as one
published last year.

## Where the numbers come from

The rule that governs the whole tool: **the language model writes prose, never
numbers.** Parameter counts, licence, context length, file sizes and every part
of the memory arithmetic are read from `config.json`, the repo's metadata and
its file listing. If the language model ever contradicts one of those, its
version is dropped.

Memory is reported in parts, never as one opaque figure:

```
weights + KV cache + a runtime allowance = total
```

The runtime allowance is an assumption rather than a measurement, and it is the
least certain number in the calculation. It is printed on its own line, named as
an assumption, and can be changed with `runtimeOverheadBytes` in the config file.

## Installing

```
npm install
npm run build
npm link        # puts `catalog` on the PATH, runnable from anywhere
```

`catalog query` also needs a language model to write the prose with. It talks to
an ollama already installed on the machine, and never installs or downloads one
itself:

```
ollama pull qwen3:8b
```

Any model you already have works instead — name it with `CATALOG_MODEL` or the
`model` setting in the config file. `catalog fit`, `specs` and `cache` involve no
language model and need none of this.

## Commands

| Command | What it does |
|---|---|
| `catalog query <url>` | What the model takes in, gives back and is for, then whether this machine runs it |
| `catalog fit <url>` | Whether this machine can run a model, with the arithmetic shown — no language model involved |
| `catalog specs` | What the tool detected about this machine |
| `catalog cache <list\|clear>` | The saved results |

Run `catalog <command> --help` for the options.

## Caching

Results are keyed by the repo's **revision SHA**, never by its name. Vendors edit
model cards in place, and a name-keyed cache would pin the first answer forever.
A new commit is a new key, so an edited card is reprocessed automatically with no
staleness check anywhere.

## Where the prose is written

`catalog query` needs a language model to write the description with. There are
two places it can run, and the choice is about privacy before it is about speed.

**On this machine, through ollama.** The default. The model card never leaves the
machine.

```
ollama pull qwen3:8b
```

**Over the network, through OpenRouter.** Nothing to install and nothing to fit
in VRAM, but the model card is posted to a third party and to whichever provider
they route it to.

```
export OPENROUTER_API_KEY=sk-or-...
CATALOG_RUNTIME=openrouter catalog query <url>
```

The key is read from the environment only, never from the config file: a key in
a file is a key that gets committed or pasted into a bug report. Switching
runtime moves the endpoint and the default model with it, so naming the runtime
is enough. `CATALOG_MODEL` overrides the model either way, and a name OpenRouter
does not serve is reported before the card is sent, not after.

The rule about numbers does not change with the runtime. Whichever model writes
the prose, it writes prose only, and every claim is still checked against the
repo's own data before it is printed.

## Configuration

`~/.config/catalog/config.json` holds the settings, including which language
model to use and which runtime serves it. Nothing in the code assumes a model
size or a hardware class.

`~/.config/catalog/machine.json` overrides anything the hardware probe reads
wrongly. Its keys match the field names in `catalog specs --json`. No probe ever
blocks a run: a field that cannot be read is reported as unknown rather than
filled in with a plausible number.

## Development

```
npm install
npm test          # never touches the network
npm run typecheck
npm run fixtures:refresh   # re-record the Hugging Face responses, by hand
```

Tests run against recorded Hugging Face responses committed under
`test/fixtures/`. A suite that depends on a live API fails for reasons unrelated
to the code and gets ignored within a week, so refreshing the fixtures is a
separate, deliberate step.

Deterministic code gets ordinary pass/fail tests. The language model's wording
does not: that belongs to the eval set, which is graded by a person.
