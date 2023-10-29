import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

// Create a read-only string widget
function createWidget(app, node, widgetName, type) {
    const widget = ComfyWidgets[type](node, widgetName, ["STRING", {multiline: true}], app).widget;
    widget.inputEl.readOnly = true;
    widget.inputEl.style.textAlign = "center";
    widget.inputEl.style.fontSize = "0.75rem";
    return widget;
}

app.registerExtension({
    name: "sd_prompt_reader.parameterDisplay",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SDParameterGenerator") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                // Create widgets
                const steps_display = createWidget(app, this, "steps_display", "STRING");
                const aspect_ratio_display = createWidget(app, this, "aspect_ratio_display", "STRING");

                // Resize the node
                const nodeWidth = this.size[0];
                const nodeHeight = this.size[1];
                this.setSize([nodeWidth * 2, nodeHeight * 1.2]);
                return result;

            };

            // Update widgets
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                let ar_message;
                if (message.text[0] === "custom") {
                    ar_message = "Custom aspect ratio: " + message.text[2] + " x " + message.text[3];
                } else {
                    ar_message = `Optimal resolution for ${message.text[1]} model
with aspect ratio ${message.text[0]}: ${message.text[2]} x ${message.text[3]}`;
                    this.widgets.find(obj => obj.name === "width").value = message.text[2];
                    this.widgets.find(obj => obj.name === "height").value = message.text[3];
                }

                const start_at_float = parseFloat(message.text[5])
                const base_percentage = Math.round(start_at_float * 100) + "%";
                // const refiner_percentage = Math.round((1 - start_at_float) * 100) + "%";
                let step_message;
                if (start_at_float === 1) {
                    step_message = `Total steps: ${message.text[4]},
Refiner off`;
                } else {
                    step_message = `Total steps: ${message.text[4]},
Refiner start at step: ${message.text[6]} (${base_percentage})`;
                }

                this.widgets.find(obj => obj.name === "steps_display").value = step_message;
                this.widgets.find(obj => obj.name === "aspect_ratio_display").value = ar_message;
            };
        }
    },
});