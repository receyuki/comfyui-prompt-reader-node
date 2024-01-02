import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";
import {createTextWidget} from "./utils.js"

// Displays prompt and setting on the node
app.registerExtension({
    name: "sd_prompt_reader.promptDisplay",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SDPromptReader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                // Create prompt and setting widgets
                const styles = {opacity: 0.7}
                const positive = createTextWidget(app, this, "positive", styles);
                const negative = createTextWidget(app, this, "negative", styles);
                const setting = createTextWidget(app, this, "setting", styles);
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
                this.widgets.find(obj => obj.name === "positive").value = message.text[0];
                this.widgets.find(obj => obj.name === "negative").value = message.text[1];
                this.widgets.find(obj => obj.name === "setting").value = message.text[2];
            };
        }
    },
});