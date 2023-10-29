import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

// Create a read-only string widget with opacity set
function createWidget(app, node, widgetName) {
    const widget = ComfyWidgets["STRING"](node, widgetName, ["STRING", {multiline: true}], app).widget;
    widget.inputEl.readOnly = true;
    widget.inputEl.style.opacity = 0.7;
    return widget;
}

// Displays prompt and setting on the node
app.registerExtension({
    name: "sd_prompt_reader.promptDisplay",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SDPromptReader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                // Create prompt and setting widgets
                const positive = createWidget(app, this, "positive");
                const negative = createWidget(app, this, "negative");
                const setting = createWidget(app, this, "setting");
                // Resize the node
                const nodeWidth = this.size[0];
                const nodeHeight = this.size[1];
                this.setSize([nodeWidth, nodeHeight * 3]);
                return result;
            };

            // Update widgets
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                this.widgets[3].value = message.text[0];
                this.widgets[4].value = message.text[1];
                this.widgets[5].value = message.text[2];
            };
        }
    },
});