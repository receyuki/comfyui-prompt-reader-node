import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";
import {createTextWidget} from "./utils.js"

// Displays file list on the node
app.registerExtension({
    name: "sd_prompt_reader.loaderDisplay",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SDBatchLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                // Create prompt and setting widgets
                const styles = {opacity: 0.7}
                const fileList =createTextWidget(app, this, "fileList", styles);
                return result;
            };

            // Update widgets
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                this.widgets.find(obj => obj.name === "fileList").value = message.text[0];

            };
        }
    },
});