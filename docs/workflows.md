# ComfyUI workflows

Run the commands from the add-on folder. Keep your exported graphs, scenes and operator state in a separate working folder. They can contain prompts, local paths and run ownership keys.

## Export a local workflow

Open **Templates** in ComfyUI and choose a local text-to-video graph for the model you use. You can also open a workflow downloaded from its publisher. [ComfyUI template browser](https://docs.comfy.org/interface/features/template)

| Model | Local graph source | Weights and setup |
| --- | --- | --- |
| LTX-2.5 | [Comfy Org LTX-2.5 text-to-video template](https://github.com/Comfy-Org/workflow_templates/blob/c067929d7e9cc0f3fd15be348fa1a1d85da15946/templates/video_ltx2_5_t2v.json) | [LTX-2.5 setup](setup.md#ltx-25) |
| LTX-2.3 | [Comfy Org LTX-2.3 text-to-video template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_ltx2_3_t2v.json) | [LTX-2.3 setup](setup.md#ltx-23) |
| MiniMax H3 | [Comfy Org MiniMax H3 text-to-video template](https://github.com/Comfy-Org/workflow_templates/blob/e7cd011d4ded3411c2f481200544f0be6fdc962e/templates/video_minimax_h3_t2v.json) | [MiniMax H3 setup](setup.md#minimax-h3) |

1. Select the weights installed on your server in the graph's loader nodes. Select the text-to-video branch if the graph also has image inputs.
2. Use **Export (API)** in ComfyUI and save the JSON outside the add-on folder. The installer reads the execution graph from this export. Choose a local template whose nodes load weights from your ComfyUI installation.
3. Register the API file once per model. Use `--workflow-api` with the file path and `--model-version` with `2.5`, `2.3` or `h3`. The importer makes the supported templates' prompt, size and duration controls editable for each scene and checks the local weights and video output. [Install and import](../README.md#2-install-the-add-on)

Native registration saves a local copy of the imported graph. To use a different template for future scenes, register its API export and approve a new scene plan. For [Docker](setup.md#docker-and-the-api), keep the exports in the mounted data folder so the container can register them on startup.

Workflows and model weights are obtained separately under their publishers' terms. Comfy Org's workflow repository uses the [MIT license](https://github.com/Comfy-Org/workflow_templates/blob/c067929d7e9cc0f3fd15be348fa1a1d85da15946/LICENSE).

## Executable request

To build a standalone operator profile and request, run the following with your ComfyUI folder and exported graph. Use the same model roots and endpoint as your installation.

```text
python configure_ltx.py --comfyui "../ComfyUI" --workflow-api "../ltx-2.5.api.json" --model-version 2.5 --output "../scene-settings"
python runtime/scripts/ltx_operator.py preflight --template-request "../scene-settings/request.json" --json
```

The settings folder contains the imported `workflow.json`, its ComfyUI `profile.json` and the executable `request.json`. The request stores the graph under `api_prompt` and records local model paths and editable parameters. A hand-edited request must also keep its media, weight requirements, displayed parameters and output paths aligned with the graph. Active weight loaders need a supported file selector. When a request has several `SaveVideo` nodes, set `variant.primary_output_node` to the string ID that saves the finished scene.

After installing the add-on and passing preflight, register the request in MPT and restart MPT. The installer checks the request and its referenced files before updating the model selection.

```text
python install.py configure "../MoneyPrinterTurbo" --request-template "../scene-settings/request.json"
```

After restarting MPT, choose **Local ComfyUI** under **Video Source**, then choose the registered model under **Local video model**. Requests without a model version appear as **Configured workflow**.

## Register a workflow for inspection

To inspect a graph by name in the operator CLI, register the imported graph in a local registry. This creates a copy of the graph and a manifest with its hash. Use the graph and profile from the executable request above.

```text
python runtime/scripts/ltx_operator.py register-workflow --source "../scene-settings/workflow.json" --family t2v-i2v-two-stage --name studio-shot --registry-root "../scene-registry"
python runtime/scripts/ltx_operator.py preflight --variant studio-shot --registry-root "../scene-registry" --profile "../scene-settings/profile.json" --json
```

## Plan and run

For a scene from the command line, save a description as `../scene-settings/overrides.json`.

```json
{"prompt": "A red paper boat glides across still water.", "output_prefix": "studio-shot"}
```

```text
python runtime/scripts/ltx_operator.py prepare-request --template-request "../scene-settings/request.json" --overrides "../scene-settings/overrides.json" --output "../scene-settings/scene.json"
python runtime/scripts/ltx_operator.py preflight --request "../scene-settings/scene.json" --json
python runtime/scripts/ltx_operator.py plan --request "../scene-settings/scene.json" --output "../scene-settings/plan.json"
```

Review the saved plan, including its scene text, executable graph, inputs, dimensions and outputs. `prepare-request` also accepts `negative_prompt` for workflows with negative guidance, plus `seeds`, `frame_count`, `width`, `height` and `client_id` for editable fields. Direct dimensions must be multiples of 32. LTX accepts 64–16,384 pixels per axis, `8n+1` frames (1–16,377), and 1–120 fps. These dimensions describe the first pass; two-stage workflows upscale them 2×. MiniMax H3 accepts 32–16,384 pixels per axis and `17n+5` frames (5–3,592) at 24 fps. Seeds range from 0 to 18,446,744,073,709,551,615.

Replace `HASH` below with the `approval_hash` of the plan you reviewed. `verify-plan` checks its current dependencies without spending the approval. Call `run` once when ready to submit, using a new run directory.

```text
python runtime/scripts/ltx_operator.py verify-plan --plan "../scene-settings/plan.json" --approval-hash HASH
python runtime/scripts/ltx_operator.py run --plan "../scene-settings/plan.json" --approval-hash HASH --run-dir "../scene-settings/run-01" --state-root "../scene-settings/state"
```

A failed preflight before submission leaves the approval unused. After the operator consumes an approval, the same hash cannot submit another job. If submission is uncertain, inspect the run before planning a retry. The run report lists the output paths. Keep the same state directory for `status` and `cancel`. Use the signed `run.json` from that run.

`run --timeout SECONDS` limits startup, preflight, submission and completion together. If a submitted job times out, checking and cancelling that owned job may take up to one additional second. The run report records whether cancellation was confirmed. A timeout never authorizes another submission.

```text
python runtime/scripts/ltx_operator.py status --run-record "../scene-settings/run-01/run.json" --state-root "../scene-settings/state"
```

Use `cancel` with the same arguments as `status` to cancel that job. ComfyUI versions with per-job cancellation support can also stop it while running. On older servers the operator can remove a pending job from the queue.

Approval includes the resolved names, paths, sizes and modification times of weights and LoRAs. Verify the file contents when downloading or replacing weights. Recreate plans made with the old weight-hashing format before approval.

## Batch planning

Save a manifest as `../scene-settings/batch.json`. Each `request_file` must be an existing absolute path. Supply the path to your own scene file in the example.

```json
{
  "schema_version": 1,
  "scenes": [
    {"scene_index": 1, "request_file": "/path/to/scene-settings/scene.json"}
  ]
}
```

```text
python runtime/scripts/ltx_operator.py plan-batch --manifest "../scene-settings/batch.json" --output-dir "../scene-settings/plans" --json
```

The CLI checks each request against the same server, workflow and dependency set, then writes the plans ready for review in scene order to a new directory. Each scene gets its own approval hash. With `--json`, stderr carries progress events and stdout contains the final summary. In MPT, the operator submits approved scenes in order with at most two outstanding. The [WebUI](../README.md#3-make-your-video) and [REST API](setup.md#docker-and-the-api) provide the review and submission steps.
