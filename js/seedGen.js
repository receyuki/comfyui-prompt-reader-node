/**
 * Modified from: https://github.com/rgthree/rgthree-comfy/blob/main/web/comfyui/seed.js
 * Modified by: receyuki
 */
import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

const LAST_SEED_BUTTON_LABEL = "(Use last queued seed)";
const SPECIAL_SEED_RANDOM = -1;
const SPECIAL_SEED_INCREMENT = -2;
const SPECIAL_SEED_DECREMENT = -3;
const SPECIAL_SEEDS = [SPECIAL_SEED_RANDOM, SPECIAL_SEED_INCREMENT, SPECIAL_SEED_DECREMENT];

function getResolver(timeout = 5000) {
    const resolver = {};
    resolver.id = generateId(8);
    resolver.completed = false;
    resolver.resolved = false;
    resolver.rejected = false;
    resolver.promise = new Promise((resolve, reject) => {
        resolver.reject = () => {
            resolver.completed = true;
            resolver.rejected = true;
            reject();
        };
        resolver.resolve = (data) => {
            resolver.completed = true;
            resolver.resolved = true;
            resolve(data);
        };
    });
    resolver.timeout = setTimeout(() => {
        if (!resolver.completed) {
            resolver.reject();
        }
    }, timeout);
    return resolver;
}

function generateId(length) {
    const arr = new Uint8Array(length / 2);
    crypto.getRandomValues(arr);
    return Array.from(arr, dec2hex).join('');
}

function dec2hex(dec) {
    return dec.toString(16).padStart(2, "0");
}

let graphResolver = null;

function waitForGraph() {
    if (graphResolver === null) {
        graphResolver = getResolver();

        function _wait() {
            if (!graphResolver.completed) {
                if (app === null || app === void 0 ? void 0 : app.graph) {
                    graphResolver.resolve(app.graph);
                } else {
                    requestAnimationFrame(_wait);
                }
            }
        }

        _wait();
    }
    return graphResolver.promise;
}

class Rgthree extends EventTarget {
    constructor() {
        super();
        this.processingQueue = false;
        this.initialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff = null;
        this.initializeGraphAndCanvasHooks();
        this.initializeComfyUIHooks();
    }

    async initializeGraphAndCanvasHooks() {
        const rgthree = this;
        const [graph] = await Promise.all([waitForGraph()]);
        const onSerialize = graph.onSerialize;
        graph.onSerialize = (data) => {
            this.initialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff = data;
            onSerialize === null || onSerialize === void 0 ? void 0 : onSerialize.call(graph, data);
        };
    }

    initializeComfyUIHooks() {
        const rgthree = this;
        const queuePrompt = app.queuePrompt;
        app.queuePrompt = async function () {
            rgthree.dispatchEvent(new CustomEvent("queue"));
            rgthree.processingQueue = true;
            try {
                await queuePrompt.apply(app, [...arguments]);
            } finally {
                rgthree.processingQueue = false;
                rgthree.dispatchEvent(new CustomEvent("queue-end"));
            }
        };
    }

    getNodeFromInitialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff(node) {
        var _a, _b, _c;
        return ((_c = (_b = (_a = this.initialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff) === null || _a === void 0 ? void 0 : _a.nodes) === null || _b === void 0 ? void 0 : _b.find((n) => n.id === node.id)) !== null && _c !== void 0 ? _c : null);
    }
}

const rgthree = new Rgthree();

class SeedControl {
    constructor(node) {
        this.lastSeed = undefined;
        this.serializedCtx = {};
        this.lastSeedValue = null;
        this.node = node;
        this.node.constructor.exposedActions = ["Randomize seed each time", "Use last queued seed"];
        const handleAction = this.node.handleAction;
        this.node.handleAction = async (action) => {
            handleAction && handleAction.call(this.node, action);
            if (action === "Randomize each time") {
                this.seedWidget.value = SPECIAL_SEED_RANDOM;
            } else if (action === "Use last queued seed") {
                this.seedWidget.value = this.lastSeed != null ? this.lastSeed : this.seedWidget.value;
                this.lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
                this.lastSeedButton.disabled = true;
            }
        };
        this.node.properties = this.node.properties || {};
        for (const [i, w] of this.node.widgets.entries()) {
            if (w.name === "seed") {
                this.seedWidget = w;
                this.seedWidget.value = SPECIAL_SEED_RANDOM;
            } else if (w.name === "control_after_generate") {
                this.node.widgets.splice(i, 1);
            }
        }
        if (!this.seedWidget) {
            throw new Error("Something's wrong; expected seed widget");
        }
        const randMax = Math.min(1125899906842624, this.seedWidget.options.max);
        const randMin = Math.max(0, this.seedWidget.options.min);
        const randomRange = (randMax - Math.max(0, randMin)) / (this.seedWidget.options.step / 10);
        this.node.addWidget("button", "Randomize seed each time", null, () => {
            this.seedWidget.value = SPECIAL_SEED_RANDOM;
        }, {serialize: false});
        this.node.addWidget("button", "New fixed random seed", null, () => {
            this.seedWidget.value =
                Math.floor(Math.random() * randomRange) * (this.seedWidget.options.step / 10) + randMin;
        }, {serialize: false});
        this.lastSeedButton = this.node.addWidget("button", LAST_SEED_BUTTON_LABEL, null, () => {
            this.seedWidget.value = this.lastSeed != null ? this.lastSeed : this.seedWidget.value;
            this.lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
            this.lastSeedButton.disabled = true;
        }, {serialize: false});
        this.lastSeedButton.disabled = true;
        this.seedWidget.serializeValue = async (node, index) => {
            const inputSeed = this.seedWidget.value;
            if (!rgthree.processingQueue) {
                return inputSeed;
            }
            this.serializedCtx = {
                inputSeed: this.seedWidget.value,
            };
            if (SPECIAL_SEEDS.includes(this.serializedCtx.inputSeed)) {
                if (typeof this.lastSeed === "number" && !SPECIAL_SEEDS.includes(this.lastSeed)) {
                    if (inputSeed === SPECIAL_SEED_INCREMENT) {
                        this.serializedCtx.seedUsed = this.lastSeed + 1;
                    } else if (inputSeed === SPECIAL_SEED_DECREMENT) {
                        this.serializedCtx.seedUsed = this.lastSeed - 1;
                    }
                }
                if (!this.serializedCtx.seedUsed || SPECIAL_SEEDS.includes(this.serializedCtx.seedUsed)) {
                    this.serializedCtx.seedUsed =
                        Math.floor(Math.random() * randomRange) * (this.seedWidget.options.step / 10) + randMin;
                }
            } else {
                this.serializedCtx.seedUsed = this.seedWidget.value;
            }
            const n = rgthree.getNodeFromInitialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff(node);
            if (n) {
                n.widgets_values[index] = this.serializedCtx.seedUsed;
            } else {
                console.warn('No serialized node found in workflow. May be attributed to '
                    + 'https://github.com/comfyanonymous/ComfyUI/issues/2193');
            }
            this.seedWidget.value = this.serializedCtx.seedUsed;
            this.lastSeed = this.serializedCtx.seedUsed;
            if (SPECIAL_SEEDS.includes(this.serializedCtx.inputSeed)) {
                this.lastSeedButton.name = `${this.serializedCtx.seedUsed}`;
                this.lastSeedButton.disabled = false;
                if (this.lastSeedValue) {
                    this.lastSeedValue.value = `Last Seed: ${this.serializedCtx.seedUsed}`;
                }
            } else {
                this.lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
                this.lastSeedButton.disabled = true;
            }
            return this.serializedCtx.seedUsed;
        };
        this.seedWidget.afterQueued = () => {
            if (this.serializedCtx.inputSeed) {
                this.seedWidget.value = this.serializedCtx.inputSeed;
            }
            this.serializedCtx = {};
        };
    }
}

app.registerExtension({
    name: "sd_prompt_reader.seedGen",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "SDParameterGenerator") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                this.seedControl = new SeedControl(this);
            };
        }
    },
});
