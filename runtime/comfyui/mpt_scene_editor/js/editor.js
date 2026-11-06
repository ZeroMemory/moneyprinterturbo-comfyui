import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const session = new URLSearchParams(window.location.search).get("mpt_edit");
const tabId = "MPT.SceneEditor";
const ownershipKey = "moneyPrinterTurboScene";
const hashPattern = /^[0-9a-f]{64}$/;
const bindings = new Map();
let manifest = null;
let panel = null;
let busy = false;
let pendingLoad = null;
let loadGeneration = 0;
let message = "Open a scene batch from MoneyPrinterTurbo.";

function object(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
}

function apiPrompt(value) {
    return object(value) && Object.keys(value).length > 0 && Object.values(value).every(
        node => object(node) && typeof node.class_type === "string" && node.class_type.length > 0
            && object(node.inputs)
    );
}

function ownership(extra = app.rootGraph.extra) {
    const tag = extra?.[ownershipKey];
    return object(tag) && tag.schema_version === 1 && /^[0-9a-f]{32}$/.test(tag.session)
        && hashPattern.test(tag.revision) && Number.isSafeInteger(tag.index) && tag.index > 0
        && hashPattern.test(tag.base_approval_hash) && typeof tag.token === "string" ? tag : undefined;
}

function matchesScene(tag, scene) {
    return tag?.session === session && tag.revision === manifest.revision && tag.index === scene.index
        && tag.base_approval_hash === scene.base_approval_hash;
}

function validateManifest(value) {
    if (!object(value) || value.schema_version !== 1 || value.session !== session
        || !hashPattern.test(value.revision) || !Array.isArray(value.scenes) || !value.scenes.length) {
        throw new Error("Invalid MPT scene manifest. Reopen the batch from MPT.");
    }
    const indexes = new Set();
    for (const scene of value.scenes) {
        if (!object(scene) || !Number.isSafeInteger(scene.index) || scene.index < 1
            || indexes.has(scene.index) || !hashPattern.test(scene.base_approval_hash)
            || typeof scene.title !== "string" || !scene.title || !apiPrompt(scene.api_prompt)) {
            throw new Error("Invalid MPT scene manifest. Reopen the batch from MPT.");
        }
        indexes.add(scene.index);
    }
    return value;
}

async function readJson(name, optional = false) {
    const response = await api.getUserData(`mpt-scenes/${session}/${name}`);
    if (optional && response.status === 404) return undefined;
    if (!response.ok) throw new Error(`Cannot read MPT scenes (HTTP ${response.status}).`);
    return response.json();
}

async function checkRevision() {
    const current = validateManifest(await readJson("manifest.json"));
    if (current.revision !== manifest.revision || current.scenes.length !== manifest.scenes.length
        || current.scenes.some(scene => !manifest.scenes.some(
            original => original.index === scene.index && original.base_approval_hash === scene.base_approval_hash
        ))) {
        throw new Error("This batch changed in MPT. Reopen it before saving edits.");
    }
}

function activeBinding() {
    const workflow = app.extensionManager.workflow.activeWorkflow;
    return workflow ? bindings.get(workflow.instanceId) : undefined;
}

function adoptRestoredScene() {
    const tag = ownership();
    const workflow = app.extensionManager.workflow.activeWorkflow;
    const scene = manifest.scenes.find(item => matchesScene(tag, item));
    if (!scene || !workflow || !matchesScene(ownership(workflow.activeState?.extra), scene)
        || ownership(workflow.activeState?.extra).token !== tag.token) return;
    const binding = { scene, workflow, path: workflow.path };
    bindings.set(workflow.instanceId, binding);
    return binding;
}

function captureScene() {
    const workflow = app.extensionManager.workflow.activeWorkflow;
    const binding = activeBinding();
    if (!binding || binding.path !== workflow.path) {
        throw new Error("Select an MPT scene before saving. Other workflows cannot replace a scene.");
    }
    return { workflow, binding, graphState: app.rootGraph.state };
}

function unchanged(capture) {
    const workflow = app.extensionManager.workflow.activeWorkflow;
    if (workflow?.instanceId !== capture.workflow.instanceId || workflow.path !== capture.binding.path
        || app.rootGraph.state !== capture.graphState) {
        throw new Error("The active workflow changed. Select the MPT scene and save it again.");
    }
}

async function saveScene() {
    const capture = captureScene();
    await checkRevision();
    unchanged(capture);
    const { workflow, output } = await app.graphToPrompt();
    unchanged(capture);
    if (!apiPrompt(output)) throw new Error("The scene has no valid API graph. Restore its missing nodes before saving.");
    const scene = capture.binding.scene;
    const envelope = {
        schema_version: 1,
        session,
        revision: manifest.revision,
        index: scene.index,
        base_approval_hash: scene.base_approval_hash,
        api_prompt: output,
        workflow,
    };
    const response = await api.storeUserData(`mpt-scenes/${session}/scene-${String(scene.index).padStart(3, "0")}.json`, envelope);
    if (!response.ok) throw new Error(`Cannot save the scene to MPT (HTTP ${response.status}).`);
    message = "Edits are saved to MPT. Start the batch from MPT.";
}

function validateSaved(value, scene) {
    if (!object(value) || value.schema_version !== 1 || value.session !== session
        || value.revision !== manifest.revision || value.index !== scene.index
        || value.base_approval_hash !== scene.base_approval_hash || !apiPrompt(value.api_prompt)
        || (value.workflow !== undefined && (!object(value.workflow)
            || !Array.isArray(value.workflow.nodes) || !Array.isArray(value.workflow.links)))) {
        throw new Error(`Saved edits for ${scene.title} do not match this batch. Reopen the batch from MPT.`);
    }
    return value;
}

async function openScene(scene) {
    const active = activeBinding() ?? adoptRestoredScene();
    if (active) {
        if (active.scene.index === scene.index) {
            captureScene();
            message = `${scene.title} is open. Save your edits before starting the batch from MPT.`;
            return;
        }
        await saveScene();
    }
    await checkRevision();
    let existing = [...bindings.values()].find(binding => binding.scene.index === scene.index
        && app.extensionManager.workflow.openWorkflows.some(workflow => workflow.instanceId === binding.workflow.instanceId));
    if (!existing) {
        const restored = app.extensionManager.workflow.openWorkflows.filter(workflow => matchesScene(ownership(workflow.activeState?.extra), scene));
        if (restored.length > 1) throw new Error("More than one restored tab contains this scene. Select the intended native tab first.");
        if (restored.length === 1) existing = { scene, workflow: restored[0], path: restored[0].path };
    }
    const saved = existing ? undefined : await readJson(`scene-${String(scene.index).padStart(3, "0")}.json`, true);
    if (saved !== undefined) validateSaved(saved, scene);
    let name = `MPT ${session} - Scene ${scene.index}`;
    for (let suffix = 2; app.extensionManager.workflow.getWorkflowByPath(`workflows/${name}.json`); suffix++) {
        name = `MPT ${session} - Scene ${scene.index} (${suffix})`;
    }
    const load = {
        tag: { schema_version: 1, session, revision: manifest.revision, index: scene.index,
            base_approval_hash: scene.base_approval_hash, token: crypto.randomUUID() },
        path: existing?.path ?? `workflows/${name}.json`,
        instanceId: existing?.workflow.instanceId,
        generation: null,
        graphState: null,
        target: null,
        cancelled: false,
    };
    pendingLoad = load;
    try {
        let workflow;
        if (existing) {
            if (existing.workflow.path !== existing.path) throw new Error("The scene workflow was renamed. Reopen this batch from MPT.");
            workflow = await app.loadGraphData(structuredClone(existing.workflow.activeState), true, true, existing.workflow);
        } else if (saved?.workflow) {
            workflow = await app.loadGraphData(structuredClone(saved.workflow), true, true, name);
        } else {
            await app.loadApiJson(structuredClone(saved?.api_prompt ?? scene.api_prompt), name);
            workflow = load.target;
        }
        if (load.cancelled || load.generation !== loadGeneration || !workflow || workflow !== load.target
            || workflow.path !== load.path || app.extensionManager.workflow.activeWorkflow?.instanceId !== workflow.instanceId
            || app.rootGraph.state !== load.graphState || ownership()?.token !== load.tag.token) {
            throw new Error("Another workflow replaced the scene while it was opening. Select the MPT scene again; no foreign tab was linked.");
        }
        bindings.set(workflow.instanceId, { scene, workflow, path: workflow.path });
        message = `${scene.title} is open. Save your edits before starting the batch from MPT.`;
    } finally {
        pendingLoad = null;
    }
}

async function run(action) {
    if (busy) return;
    busy = true;
    renderPanel();
    try {
        await action();
    } catch (error) {
        message = error instanceof Error ? error.message : String(error);
        app.extensionManager.toast.error("MPT scene editor", { description: message });
    } finally {
        busy = false;
        renderPanel();
    }
}

function renderPanel() {
    if (!panel) return;
    panel.replaceChildren();
    const heading = document.createElement("h2");
    heading.textContent = "MPT scenes";
    const instructions = document.createElement("p");
    instructions.textContent = "Choose a scene, edit its workflow, and save it to MPT. Switching scenes saves the scene you are leaving.";
    panel.append(heading, instructions);
    const selected = activeBinding();
    for (const scene of manifest?.scenes ?? []) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = scene.title;
        button.disabled = busy;
        button.setAttribute("aria-pressed", String(selected?.scene.index === scene.index));
        button.style.cssText = "display:block;width:100%;margin:0.5rem 0;padding:0.5rem;text-align:left;border:1px solid var(--border-color);border-radius:0.25rem;";
        button.style.fontWeight = selected?.scene.index === scene.index ? "bold" : "normal";
        button.addEventListener("click", () => void run(() => openScene(scene)));
        panel.append(button);
    }
    const save = document.createElement("button");
    save.type = "button";
    save.textContent = busy ? "Working…" : "Save scene to MPT";
    save.disabled = busy || !selected;
    save.style.cssText = "width:100%;padding:0.5rem;border:1px solid var(--border-color);border-radius:0.25rem;";
    save.addEventListener("click", () => void run(saveScene));
    const status = document.createElement("p");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    status.textContent = message;
    panel.append(save, status);
}

function showScenes() {
    app.extensionManager.command.execute(`Workspace.ToggleSidebarTab.${tabId}`);
}

app.registerExtension({
    name: "MoneyPrinterTurbo.SceneEditor",
    commands: [
        { id: "MPT.ShowScenes", label: "MPT scenes", function: showScenes },
        { id: "MPT.SaveScene", label: "Save scene to MPT", function: () => run(saveScene) },
    ],
    menuCommands: [{ path: ["MoneyPrinterTurbo"], commands: ["MPT.ShowScenes", "MPT.SaveScene"] }],
    actionBarButtons: [{ icon: "icon-[lucide--list]", label: "MPT scenes", onClick: showScenes }],
    async setup() {
        app.extensionManager.registerSidebarTab({
            id: tabId,
            title: "MPT scenes",
            tooltip: "Edit scenes from MoneyPrinterTurbo",
            icon: "icon-[lucide--list]",
            type: "custom",
            render(container) {
                panel = container;
                panel.style.cssText = "padding:1rem;overflow:auto;";
                renderPanel();
            },
            destroy() { panel = null; },
        });
        const queuePrompt = app.queuePrompt;
        app.queuePrompt = async function (...args) {
            if (activeBinding() || ownership()) {
                message = "Save your scene edits to MPT. Start the batch from MPT.";
                renderPanel();
                app.extensionManager.toast.warning("Start this batch from MPT", { description: message });
                return false;
            }
            return queuePrompt.apply(this, args);
        };
        const submitPrompt = api.queuePrompt;
        api.queuePrompt = async function (number, data, ...args) {
            if (ownership(data?.workflow?.extra)) {
                message = "Save your scene edits to MPT. Start the batch from MPT.";
                renderPanel();
                throw new Error(message);
            }
            return submitPrompt.call(this, number, data, ...args);
        };
        const dispatch = api.dispatchCustomEvent;
        api.dispatchCustomEvent = function (type, ...args) {
            if (type === "autoQueueGraphChanged" && (activeBinding() || ownership())) return true;
            return dispatch.call(this, type, ...args);
        };
        if (session === null) return;
        if (!/^[0-9a-f]{32}$/.test(session)) {
            message = "Invalid MPT edit session. Open the scene batch from MPT again.";
            renderPanel();
            app.extensionManager.toast.error("MPT scene editor", { description: message });
            return;
        }
        showScenes();
        await run(async () => {
            manifest = validateManifest(await readJson("manifest.json"));
            message = "Select a scene to edit. Generation starts only from MPT.";
        });
    },
    beforeLoadGraph() {
        loadGeneration++;
        if (!pendingLoad) return;
        if (pendingLoad.generation === null) pendingLoad.generation = loadGeneration;
        else pendingLoad.cancelled = true;
    },
    afterConfigureGraph() {
        if (!pendingLoad || pendingLoad.cancelled) return;
        if (pendingLoad.generation !== loadGeneration || pendingLoad.graphState !== null) {
            pendingLoad.cancelled = true;
            return;
        }
        pendingLoad.graphState = app.rootGraph.state;
        app.rootGraph.extra ??= {};
        app.rootGraph.extra[ownershipKey] = { ...pendingLoad.tag };
    },
    afterLoadGraph() {
        const load = pendingLoad;
        if (load && !load.cancelled && load.generation === loadGeneration && app.rootGraph.state === load.graphState
            && ownership()?.token === load.tag.token) {
            const workflow = app.extensionManager.workflow.activeWorkflow;
            if (workflow?.path === load.path && (!load.instanceId || workflow.instanceId === load.instanceId)
                && ownership(workflow.activeState?.extra)?.token === load.tag.token) load.target = workflow;
            else load.cancelled = true;
        }
        renderPanel();
    },
});
