<template>
    <!-- Chat Input Section -->
    <div class="chatBody">
        <!-- Chat Messages Section -->
        <div class="flex-grow overflow-y-auto chatSection">
					<ScrollPanel ref="scrollPanelRef" style="width: 100%; height: 80vh;">
            <div v-for="(msg, index) in messages" :key="index" class="mb-3">
							<Card v-if="msg.user" class="mt-2">
								<template #title> 
									<Avatar image="/Logos/User.webp" class="mr-2" size="xlarge" shape="circle" />
								</template>
								<template #content>
									<p class="m-0">
										{{ msg.text }}
									</p>
								</template>
							</Card>
							<Card v-if="!msg.user" class="chat-card mt-2">
								<template #title v-if="!msg.user" > 
									<div style="display: flex; justify-content: flex-end; align-items: center;">
										<Avatar image="/Logos/GPT-Logo.webp" class="mr-2" size="xlarge" shape="circle" />
									</div>
								</template>
								<template #content>
									<div style="display: flex; justify-content: flex-end; align-items: center;">

										<p class="m-0">
												{{ msg.text }}
										</p>
									</div>
								</template>
							</card>
            </div>
					</ScrollPanel>
        </div>
        <div class="flex items-center space-x-2 inputSection">
            <InputText
                v-model="newMessage"
                placeholder="Type your message..."
                class="flex-grow text-black inputText"
            />
						<Button icon="pi pi-arrow-up" aria-label="Save" @click="sendMessage" rounded severity="contrast" class="m-3"/>
        </div>
    </div>
</template>


<script setup>
  import { ref } from "vue";
	import Card from 'primevue/card';
	import ScrollPanel from 'primevue/scrollpanel';
	import Avatar from 'primevue/avatar';


	const scrollPanelRef = ref(null);
  const messages = ref([]);
  const newMessage = ref("");

  const sendMessage = () => {
		if (newMessage.value !== "") {
			messages.value.push({
				text: newMessage.value,
				user: true
			});
			messages.value.push({
				text: "Hola desde el robot",
				user: false
			});
			newMessage.value = ""; 
		}
	};
</script>


<style scoped>

.btnInput{
    width: 15%;
}

.inputText{
    width: 70%;
    margin-left: 11%;
}

.chatBody{
    width: 100%;
		padding-top: 5%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.chatSection{ 
  flex-grow: 1; 
  width: 100%;
}

.inputSection{
  height: 55px;
	margin: left 7%;
  width: 100%;
}

.user-card {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 8px;
}

.chat-card {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 8px;
}
</style>
